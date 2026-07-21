/*
Pure-C test — uses shared dispatcher.
nvcc -I csrc -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    csrc/tests/attn_prefill_test.cu -o test && ./test
*/

#include "test_utils.cuh"
#include "../kernels/attn_dispatchers.cuh"

// Warmed-up, CUDA-event timed throughput sweep over the production MMA path.
static void bench() {
    const int cfgs[][7] = {
        {1,32,4,512,512,128,0},
        {1,32,4,1024,1024,128,0},
        {1,32,4,2048,2048,128,0},
        {1,32,4,2048,2048,128,1},
        {4,32,4,2048,2048,128,1},
        {1,32,4,4096,4096,128,1},
    };
    int n = sizeof(cfgs)/sizeof(cfgs[0]);
    const int WARMUP = 10, ITERS = 50;
    printf("\n===== PREFILL BENCH (warmup=%d iters=%d) =====\n", WARMUP, ITERS);
    printf("%-46s | %10s | %10s | %10s\n",
           "config", "latency", "bandwidth", "throughput");
    printf("---------------------------------------------------------------"
           "----------------------------\n");

    for (int ci = 0; ci < n; ci++) {
        int B=cfgs[ci][0], Hq=cfgs[ci][1], Hk=cfgs[ci][2];
        int ql=cfgs[ci][3], kl=cfgs[ci][4], D=cfgs[ci][5], causal=cfgs[ci][6];
        size_t nQ=(size_t)B*Hq*ql*D, nKV=(size_t)B*Hk*kl*D;

        bf16 *dQ,*dK,*dV,*dO,*tmp;
        cudaMalloc(&dQ,nQ*2); cudaMalloc(&dK,nKV*2);
        cudaMalloc(&dV,nKV*2); cudaMalloc(&dO,nQ*2);
        size_t big = nQ>nKV?nQ:nKV; tmp=new bf16[big];
        for (size_t i=0;i<nQ;i++)  tmp[i]=f2bf(randf());
        cudaMemcpy(dQ,tmp,nQ*2,cudaMemcpyHostToDevice);
        for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(randf());
        cudaMemcpy(dK,tmp,nKV*2,cudaMemcpyHostToDevice);
        for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(randf());
        cudaMemcpy(dV,tmp,nKV*2,cudaMemcpyHostToDevice);

        AttentionParams<bf16> p;
        p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=ql; p.kv_len=kl; p.head_dim=D;
        p.use_mask=0; p.causal_offset=causal?0:-1;
        set_default_strides(p);
        p.scale=1.0f/sqrtf((float)D);
        p.q=dQ; p.k=dK; p.v=dV; p.mask=nullptr; p.o=dO;

        auto launch = [&]() { dispatch_by_head_dim(D, [&]<int H>() { dispatch_prefill<H>(p); }); };
        for (int i=0;i<WARMUP;i++) launch();
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return;}

        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i=0;i<ITERS;i++) launch();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=ITERS;

        double flops = 4.0*B*Hq*(double)ql*kl*D;
        if (causal) flops *= 0.5;
        double tflops = flops/(ms*1e-3)/1e12;
        double bytes = 2.0 * (2.0*nQ + 2.0*nKV);
        double gbps = bytes/(ms*1e-3)/1e9;

        char cfg[64];
        snprintf(cfg, sizeof(cfg),
                 "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d",
                 B,Hq,Hk,ql,kl,D,causal);
        printf("%-46s | %7.4f ms | %7.1f GB/s | %6.2f TFLOP/s\n",
               cfg, ms, gbps, tflops);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
        delete[]tmp; cudaEventDestroy(s); cudaEventDestroy(e);
    }
}

static int run_test(int B, int Hq, int Hk, int ql, int kl, int D, int causal) {
    printf("=== B=%d Hq=%d Hk=%d q=%d kv=%d D=%d causal=%d ===\n",
           B,Hq,Hk,ql,kl,D,causal);

    size_t nQ = B*Hq*ql*D, nKV = B*Hk*kl*D;
    float *hQ=new float[nQ], *hK=new float[nKV], *hV=new float[nKV];
    for (size_t i=0;i<nQ;i++) hQ[i]=randf();
    for (size_t i=0;i<nKV;i++){hK[i]=randf();hV[i]=randf();}

    bf16 *dQ,*dK,*dV,*dO,*tmp;
    cudaMalloc(&dQ,nQ*2); cudaMalloc(&dK,nKV*2);
    cudaMalloc(&dV,nKV*2); cudaMalloc(&dO,nQ*2);
    tmp=new bf16[max(nQ,nKV)];
    for (size_t i=0;i<nQ;i++) tmp[i]=f2bf(hQ[i]);
    cudaMemcpy(dQ,tmp,nQ*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hK[i]);
    cudaMemcpy(dK,tmp,nKV*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hV[i]);
    cudaMemcpy(dV,tmp,nKV*2,cudaMemcpyHostToDevice);

    AttentionParams<bf16> p;
    p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=ql; p.kv_len=kl; p.head_dim=D;
    p.use_mask=0; p.causal_offset=causal?0:-1;
    set_default_strides(p);
    p.scale=1.0f/sqrtf((float)D);
    p.q=dQ; p.k=dK; p.v=dV; p.mask=nullptr; p.o=dO;

    double t0=now_ms();
    dispatch_by_head_dim(D, [&]<int H>() { dispatch_prefill<H>(p); });
    cudaDeviceSynchronize();
    double kms=now_ms()-t0;
    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

    bf16* hOut=new bf16[nQ];
    cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

    float* ref=new float[nQ];
    cpu_attention_ref(hQ, hK, hV, nullptr, ref, B, Hq, Hk, ql, kl, D, causal ? 0 : -1);

    float max_abs_err=0, max_rel_err=0;
    for (size_t i=0;i<nQ;i++) {
        float err=fabsf(bf2f(hOut[i])-ref[i]);
        if(err>max_abs_err) max_abs_err=err;
        float rel=err/fmaxf(fabsf(ref[i]), 1e-8f);
        if(rel>max_rel_err) max_rel_err=rel;
    }
    const float atol=0.01f, rtol=0.01f;
    bool pass=true;
    for (size_t i=0;i<nQ;i++) {
        float err=fabsf(bf2f(hOut[i])-ref[i]);
        if (err > atol + rtol * fabsf(ref[i])) { pass=false; break; }
    }
    printf("kernel: %.3f ms  max_abs_err: %.6e  max_rel_err: %.6e  %s\n\n",
           kms, max_abs_err, max_rel_err, pass?"PASS":"FAIL");

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    delete[]hQ;delete[]hK;delete[]hV;delete[]hOut;delete[]ref;delete[]tmp;

    return pass ? 0 : 1;
}

int main() {
    const int configs[][7] = {
        {1,2,1,64,128,64,0},     // tiny: B,Hq,Hk,q,kv,D,causal
        {1,32,4,512,512,128,0},  // standard
        {1,32,4,128,256,128,0},  // medium
        {1,4,2,256,256,128,1},   // causal
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    int fail = 0;

    for (int ci = 0; ci < n_configs; ci++) {
        int B=configs[ci][0], Hq=configs[ci][1], Hk=configs[ci][2];
        int ql=configs[ci][3], kl=configs[ci][4], D=configs[ci][5];
        int causal=configs[ci][6];
        fail += run_test(B, Hq, Hk, ql, kl, D, causal);
        if (fail) break;
    }

    if (fail) {
        printf("FAILED\n");
        return fail;
    }
    printf("All tests passed!\n");
    bench();
    return 0;
}
