/*
Pure-C test:
nvcc -I csrc -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    csrc/tests/gqa_decode_test.cu -o test && ./test
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include "../kernels/gqa_decode_attn.cuh"

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void cpu_decode(const float* Q, const float* K, const float* V,
                        const bool* mask, float* O,
                        int B, int Hq, int Hk, int seq_len, int D) {
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hk;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < Hq; h++) {
            int kv_h = h / n_rep;
            float mv = -INFINITY, sv = 0.0f;
            float accum[256] = {0};
            for (int s = 0; s < seq_len; s++) {
                if (!mask[b * seq_len + s]) continue;
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q[((b * Hq + h) * 1 + 0) * D + d]
                         * K[((b * Hk + kv_h) * seq_len + s) * D + d];
                dot *= scale;
                float nm = fmaxf(mv, dot);
                float al = expf(mv - nm);
                float be = expf(dot - nm);
                sv = sv * al + be;
                for (int d = 0; d < D; d++)
                    accum[d] = accum[d] * al
                             + V[((b * Hk + kv_h) * seq_len + s) * D + d] * be;
                mv = nm;
            }
            float inv = 1.0f / sv;
            for (int d = 0; d < D; d++)
                O[((b * Hq + h) * 1 + 0) * D + d] = accum[d] * inv;
        }
    }
}

static bf16 f2bf(float x) { return __float2bfloat16(x); }
static float bf2f(bf16 x) { return __bfloat162float(x); }
static float randf() { return (float)rand() / (float)RAND_MAX - 0.5f; }

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

        GQAParams p;
        p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=1; p.kv_len=sl; p.head_dim=D;
        p.use_mask=1; p.is_causal=0; p.causal_offset=0;
        p.scale=1.0f/sqrtf((float)D);
        p.q=dQ; p.k=dK; p.v=dV; p.mask=dMask; p.o=dO;

        size_t smem=DC_CHUNK*D*sizeof(bf16);
        dim3 block(32, gs);
        dim3 grid(B*Hk);
        printf("grid=(%d,1,1) block=(%d,%d,1) smem=%zu\n",
               grid.x, block.x, block.y, smem);

        double t0=now_ms();
        gqa_decode_attn_kernel<<<grid,block,smem>>>(p);
        cudaDeviceSynchronize();
        double kms=now_ms()-t0;
        cudaError_t err=cudaGetLastError();
        if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

        bf16* hOut=new bf16[nQ];
        cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

        float* ref=new float[nQ];
        cpu_decode(hQ,hK,hV,hMask,ref,B,Hq,Hk,sl,D);

        float max_err=0;
        for (size_t i=0;i<nQ;i++){
            float d=fabsf(bf2f(hOut[i])-ref[i]);
            if(d>max_err) max_err=d;
        }
        printf("kernel: %.3f ms  max_err: %.6e\n\n",kms,max_err);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dMask);
        delete[]hQ;delete[]hK;delete[]hV;delete[]hMask;delete[]hOut;delete[]ref;delete[]tmp;
    }
    printf("All tests passed!\n");
    return 0;
}
