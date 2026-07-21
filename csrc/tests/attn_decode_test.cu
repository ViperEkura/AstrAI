/*
Pure-C test — uses shared dispatcher.
nvcc -I csrc -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    csrc/tests/attn_decode_test.cu -o test && ./test
*/

#include "test_utils.cuh"
#include "../kernels/attn_dispatchers.cuh"

// Split-K scratch (torch-free)
struct DecodeScratch {
    float* o_part = nullptr;
    float* ml_part = nullptr;
};

static void setup_scratch(AttentionParams<bf16>& p, DecodeScratch& sc) {
    int max_splits = 32;
    cudaMalloc(&sc.o_part, (size_t)p.batch * p.q_head * max_splits * p.head_dim * sizeof(float));
    cudaMalloc(&sc.ml_part, (size_t)p.batch * p.q_head * max_splits * 2 * sizeof(float));
}

static void free_scratch(DecodeScratch& sc) {
    cudaFree(sc.o_part); cudaFree(sc.ml_part);
}

// Warmed-up, CUDA-event timed sweep over the production decode MMA path.
static void bench() {
    const int cfgs[][5] = {
        {1, 32, 4, 512, 128},
        {1, 32, 4, 1024, 128},
        {1, 32, 4, 2048, 128},
        {1, 32, 4, 4096, 128},
        {16, 32, 4, 2048, 128},
        {32, 32, 4, 1024, 128},
    };
    const int WARMUP = 10, ITERS = 100;
    printf("\n===== DECODE BENCH (warmup=%d iters=%d) =====\n", WARMUP, ITERS);
    print_bench_header();

    for (int ci = 0; ci < 6; ci++) {
        int B = cfgs[ci][0], Hq = cfgs[ci][1], Hk = cfgs[ci][2];
        int sl = cfgs[ci][3], D = cfgs[ci][4];
        size_t nQ = (size_t)B * Hq * D;
        size_t nKV = (size_t)B * Hk * sl * D;

        bf16 *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, nQ*2); cudaMalloc(&dK, nKV*2);
        cudaMalloc(&dV, nKV*2); cudaMalloc(&dO, nQ*2);
        size_t big = nQ > nKV ? nQ : nKV; bf16* tmp = new bf16[big];
        for (size_t i = 0; i < nQ; i++)  tmp[i] = f2bf(randf());
        cudaMemcpy(dQ, tmp, nQ*2, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < nKV; i++) tmp[i] = f2bf(randf());
        cudaMemcpy(dK, tmp, nKV*2, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < nKV; i++) tmp[i] = f2bf(randf());
        cudaMemcpy(dV, tmp, nKV*2, cudaMemcpyHostToDevice);
        delete[] tmp;

        AttentionParams<bf16> p;
        p.batch = B; p.q_head = Hq; p.kv_head = Hk; p.q_len = 1; p.kv_len = sl;
        p.head_dim = D; p.use_mask = 0; p.causal_offset = -1;
        p.scale = 1.0f / sqrtf((float)D);
        set_default_strides(p);
        p.q = dQ; p.k = dK; p.v = dV; p.mask = nullptr; p.o = dO;

        DecodeScratch sc;
        setup_scratch(p, sc);
        p.o_part = sc.o_part; p.ml_part = sc.ml_part;

        auto launch = [&]() { dispatch_by_head_dim(D, [&]<int H>() { dispatch_decode<H>(p); }); };
        double flops = 4.0 * B * Hq * (double)sl * D;
        double bytes = 2.0 * (2.0 * nKV * sizeof(bf16));
        BenchResult r = bench_kernel(launch, WARMUP, ITERS, flops, bytes);

        char cfg[64];
        snprintf(cfg, sizeof(cfg),
                 "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d",
                 B, Hq, Hk, 1, sl, D, 0);
        print_bench_row(cfg, r);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        free_scratch(sc);
    }
}

static int run_test(int B, int Hq, int Hk, int sl, int D, int causal) {
    int gs = Hq / Hk;
    printf("=== B=%d Hq=%d Hk=%d seq=%d D=%d gs=%d causal=%d ===\n",
           B,Hq,Hk,sl,D,gs,causal);

    size_t nQ = B*Hq*1*D, nKV = B*Hk*sl*D;
    float *hQ=new float[nQ], *hK=new float[nKV], *hV=new float[nKV];
    for (size_t i=0;i<nQ;i++) hQ[i]=randf();
    for (size_t i=0;i<nKV;i++){hK[i]=randf();hV[i]=randf();}

    bool* hMask=new bool[B*sl];
    for (int i=0;i<B*sl;i++) hMask[i]=true;

    bf16 *dQ,*dK,*dV,*dO,*tmp;
    bool* dMask;
    cudaMalloc(&dQ,nQ*2); cudaMalloc(&dK,nKV*2);
    cudaMalloc(&dV,nKV*2); cudaMalloc(&dO,nQ*2);
    cudaMalloc(&dMask,B*sl);

    tmp=new bf16[max(nQ,nKV)];
    for (size_t i=0;i<nQ;i++) tmp[i]=f2bf(hQ[i]);
    cudaMemcpy(dQ,tmp,nQ*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hK[i]);
    cudaMemcpy(dK,tmp,nKV*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hV[i]);
    cudaMemcpy(dV,tmp,nKV*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dMask,hMask,B*sl,cudaMemcpyHostToDevice);

    AttentionParams<bf16> p;
    p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=1; p.kv_len=sl; p.head_dim=D;
    p.use_mask=0; p.causal_offset=causal?0:-1;
    p.scale=1.0f/sqrtf((float)D);
    set_default_strides(p);
    p.q=dQ; p.k=dK; p.v=dV; p.mask=nullptr; p.o=dO;

    DecodeScratch sc;
    setup_scratch(p, sc);
    p.o_part = sc.o_part; p.ml_part = sc.ml_part;

    double t0=now_ms();
    dispatch_by_head_dim(D, [&]<int H>() { dispatch_decode<H>(p); });
    cudaDeviceSynchronize();
    double kms=now_ms()-t0;
    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

    bf16* hOut=new bf16[nQ];
    cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

    float* ref=new float[nQ];
    cpu_attention_ref(hQ, hK, hV, hMask, ref, B, Hq, Hk, 1, sl, D, causal ? 0 : -1);

    float max_err=0;
    for (size_t i=0;i<nQ;i++){
        float d=fabsf(bf2f(hOut[i])-ref[i]);
        if(d>max_err) max_err=d;
    }
    printf("kernel: %.3f ms  max_err: %.6e\n\n",kms,max_err);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dMask);
    free_scratch(sc);
    delete[]hQ;delete[]hK;delete[]hV;delete[]hMask;delete[]hOut;delete[]ref;delete[]tmp;

    return (max_err < 0.05f) ? 0 : 1;
}

int main() {
    const int configs[][6] = {
        {1, 2, 1, 64, 32, 0},
        {1, 32, 4, 512, 128, 0},
        {1, 32, 4, 1024, 128, 0},
        {1, 32, 4, 512, 128, 1},
    };
    int n_cfgs = sizeof(configs) / sizeof(configs[0]);
    int fail = 0;

    for (int ci = 0; ci < n_cfgs; ci++) {
        int B = configs[ci][0], Hq = configs[ci][1], Hk = configs[ci][2];
        int sl = configs[ci][3], D = configs[ci][4], causal = configs[ci][5];
        fail += run_test(B, Hq, Hk, sl, D, causal);
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
