/*
Pure-C test:
nvcc -I csrc -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    csrc/tests/attn_decode_test.cu -o test && ./test
*/

#include "test_utils.cuh"
#include "../kernels/attn_decode_split_kv.cuh"
#ifndef ASTRAI_NO_MMA
#include "../kernels/attn_decode_split_kv_mma.cuh"
#endif

// Split-K scratch (torch-free): the production launcher allocates these from
// torch; here we pass pre-allocated device buffers so the bench loop doesn't
// pay a cudaMalloc per iteration. Size for the maximum split count (32).
struct DecodeScratch {
    float* o_part = nullptr;
    float* ml_part = nullptr;
};

// Launch the production decode path (tensor-core head-packing MMA on sm_80+,
// scalar fallback otherwise), mirroring dispatch_decode() in attn_decode.cu.
#ifndef ASTRAI_NO_MMA
static bool decode_use_mma(const AttentionParams<bf16>& p) {
    int G = p.q_head / p.kv_head;
    return !p.use_mask && G > 1 && G <= 16;
}

template <int HEAD_DIM, int BC>
static void launch_mma_decode(AttentionParams<bf16>& p, DecodeScratch& sc) {
    int tiles_total = (p.kv_len + BC - 1) / BC;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, tiles_total);
    p.o_part = sc.o_part;
    p.ml_part = sc.ml_part;

    attn_decode_split_kv_mma_kernel<HEAD_DIM, BC>
        <<<dim3(p.kv_head, p.batch, p.num_splits), 32>>>(p);
    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}
#endif

static void launch_scalar_decode(AttentionParams<bf16>& p, DecodeScratch& sc) {
    int gs = p.q_head / p.kv_head;
    int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    p.o_part = sc.o_part;
    p.ml_part = sc.ml_part;

    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    attn_decode_split_kv_kernel<<<dim3(p.batch * p.kv_head, 1, p.num_splits), dim3(32, gs), smem>>>(p);
    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

template <int HEAD_DIM>
static void dispatch_decode_t(AttentionParams<bf16>& p, DecodeScratch& sc) {
#ifndef ASTRAI_NO_MMA
    if (decode_use_mma(p)) { launch_mma_decode<HEAD_DIM, 32>(p, sc); return; }
#endif
    launch_scalar_decode(p, sc);
}

static void dispatch_decode(AttentionParams<bf16>& p, DecodeScratch& sc) {
    switch (p.head_dim) {
        case 32:  dispatch_decode_t<32>(p, sc);  break;
        case 64:  dispatch_decode_t<64>(p, sc);  break;
        case 128: dispatch_decode_t<128>(p, sc); break;
        case 256: dispatch_decode_t<256>(p, sc); break;
        default:  printf("bench: unsupported D=%d\n", p.head_dim);
    }
}

// Warmed-up, CUDA-event timed sweep over the production decode MMA path.
// Decode (q_len==1) is memory-bound: the two matmuls are GEMV-shaped, so we
// report both effective K/V read bandwidth and the (small) attention FLOP/s.
// FLOP/s = 2 matmuls (q@K^T, P@V), each 2*B*Hq*kv*D flops.
// Bytes    = K + V read = 2 * B*Hk*kv*D * sizeof(bf16).
static void bench() {
    const int cfgs[][5] = {
        {1, 32, 4, 512, 128},    // B,Hq,Hk,seq,D
        {1, 32, 4, 1024, 128},
        {1, 32, 4, 2048, 128},
        {1, 32, 4, 4096, 128},
        {16, 32, 4, 2048, 128},
        {32, 32, 4, 1024, 128},
    };
    int n = sizeof(cfgs)/sizeof(cfgs[0]);
    const int WARMUP = 10, ITERS = 100;
    printf("\n===== DECODE BENCH (warmup=%d iters=%d) =====\n", WARMUP, ITERS);
    printf("%-46s | %10s | %10s | %10s\n",
           "config", "latency", "bandwidth", "throughput");
    printf("---------------------------------------------------------------"
           "----------------------------\n");

    for (int ci = 0; ci < n; ci++) {
        int B=cfgs[ci][0], Hq=cfgs[ci][1], Hk=cfgs[ci][2];
        int sl=cfgs[ci][3], D=cfgs[ci][4];
        size_t nQ=(size_t)B*Hq*D, nKV=(size_t)B*Hk*sl*D;

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
        p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=1; p.kv_len=sl; p.head_dim=D;
        p.use_mask=0; p.is_causal=0; p.causal_offset=0;
        p.scale=1.0f/sqrtf((float)D);
        p.q=dQ; p.k=dK; p.v=dV; p.mask=nullptr; p.o=dO;

        DecodeScratch sc;
        cudaMalloc(&sc.o_part, (size_t)B*Hq*32*D*sizeof(float));
        cudaMalloc(&sc.ml_part, (size_t)B*Hq*32*2*sizeof(float));

        for (int i=0;i<WARMUP;i++) dispatch_decode(p, sc);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return;}

        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i=0;i<ITERS;i++) dispatch_decode(p, sc);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=ITERS;

        double flops = 4.0*B*Hq*(double)sl*D;
        double tflops = flops/(ms*1e-3)/1e12;
        // HBM traffic: K + V read (B*Hk*sl*D each), bf16; Q/O negligible.
        double bytes = 2.0 * (2.0*nKV);
        double gbps = bytes/(ms*1e-3)/1e9;

        char cfg[64];
        snprintf(cfg, sizeof(cfg),
                 "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d",
                 B,Hq,Hk,1,sl,D,0);
        printf("%-46s | %7.4f ms | %7.1f GB/s | %6.2f TFLOP/s\n",
               cfg, ms, gbps, tflops);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
        cudaFree(sc.o_part);cudaFree(sc.ml_part);
        delete[]tmp; cudaEventDestroy(s); cudaEventDestroy(e);
    }
}

int main() {
    const int configs[][5] = {
        {1, 2, 1, 64, 32},    // B,Hq,Hk,seq_len,D
        {1, 32, 4, 512, 128},
        {1, 32, 4, 1024, 128},
    };
    int n_cfgs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_cfgs; ci++) {
        int B = configs[ci][0], Hq = configs[ci][1], Hk = configs[ci][2];
        int sl = configs[ci][3], D = configs[ci][4], gs = Hq / Hk;
        printf("=== B=%d Hq=%d Hk=%d seq=%d D=%d gs=%d ===\n", B,Hq,Hk,sl,D,gs);

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
        p.use_mask=0; p.is_causal=0; p.causal_offset=0;
        p.scale=1.0f/sqrtf((float)D);
        p.q=dQ; p.k=dK; p.v=dV; p.mask=nullptr; p.o=dO;

        // Split-K scratch (max 32 splits), sized for the production MMA path.
        DecodeScratch sc;
        cudaMalloc(&sc.o_part, (size_t)B*Hq*32*D*sizeof(float));
        cudaMalloc(&sc.ml_part, (size_t)B*Hq*32*2*sizeof(float));

        double t0=now_ms();
        dispatch_decode(p, sc);
        cudaDeviceSynchronize();
        double kms=now_ms()-t0;
        cudaError_t err=cudaGetLastError();
        if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

        bf16* hOut=new bf16[nQ];
        cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

        float* ref=new float[nQ];
        cpu_attention_ref(hQ, hK, hV, hMask, ref, B, Hq, Hk, 1, sl, D, 0, 0);

        float max_err=0;
        for (size_t i=0;i<nQ;i++){
            float d=fabsf(bf2f(hOut[i])-ref[i]);
            if(d>max_err) max_err=d;
        }
        printf("kernel: %.3f ms  max_err: %.6e\n\n",kms,max_err);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dMask);
        cudaFree(sc.o_part);cudaFree(sc.ml_part);
        delete[]hQ;delete[]hK;delete[]hV;delete[]hMask;delete[]hOut;delete[]ref;delete[]tmp;
    }
    printf("All tests passed!\n");
    bench();
    return 0;
}
