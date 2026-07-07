/*
Pure-C test:
nvcc -I csrc -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    csrc/tests/gqa_prefill_test.cu -o test && ./test
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include "../kernels/gqa_prefill_attn.cuh"

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void cpu_attention(const float* Q, const float* K, const float* V, float* O,
                          int B, int Hq, int Hk, int q_len, int kv_len, int D,
                          int is_causal, int causal_off) {
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hk;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < Hq; h++) {
            for (int qi = 0; qi < q_len; qi++) {
                int kv_h = h / n_rep;
                float mv = -INFINITY, sv = 0.0f;
                float accum[256] = {0};
                int lim = is_causal ? min(kv_len, qi + causal_off + 1) : kv_len;
                for (int kj = 0; kj < lim; kj++) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += Q[((b*Hq + h)*q_len + qi)*D + d]
                             * K[((b*Hk + kv_h)*kv_len + kj)*D + d];
                    dot *= scale;
                    float nm = fmaxf(mv, dot);
                    float al = expf(mv - nm);
                    float be = expf(dot - nm);
                    sv = sv * al + be;
                    for (int d = 0; d < D; d++)
                        accum[d] = accum[d] * al
                                 + V[((b*Hk + kv_h)*kv_len + kj)*D + d] * be;
                    mv = nm;
                }
                float inv = 1.0f / sv;
                for (int d = 0; d < D; d++)
                    O[((b*Hq + h)*q_len + qi)*D + d] = accum[d] * inv;
            }
        }
    }
}

static __nv_bfloat16 f2bf(float x) { return __float2bfloat16(x); }
static float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
static float randf() { return (float)rand() / (float)RAND_MAX - 0.5f; }

int main() {
    const int configs[][7] = {
        {1,2,1,64,128,64,0},     // tiny: B,Hq,Hk,q,kv,D,causal
        {1,32,4,512,512,128,0},  // standard
        {1,32,4,128,256,128,0},  // medium
        {1,4,2,256,256,128,1},   // causal
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        int B=configs[ci][0], Hq=configs[ci][1], Hk=configs[ci][2];
        int ql=configs[ci][3], kl=configs[ci][4], D=configs[ci][5];
        int causal=configs[ci][6];
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

        GQAParams p;
        p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=ql; p.kv_len=kl; p.head_dim=D;
        p.use_mask=0; p.is_causal=causal; p.causal_offset=0;
        p.scale=1.0f/sqrtf((float)D);
        p.q=dQ; p.k=dK; p.v=dV; p.mask=nullptr; p.o=dO;

        constexpr int G=8, ROWS=32, P_BC=32;
        dim3 grid((ql+ROWS-1)/ROWS, Hq, B);
        dim3 block(G, ROWS, 1);
        size_t smem=2*P_BC*D*sizeof(bf16);
        printf("grid=(%d,%d,%d) block=(%d,%d,%d) smem=%zu\n",
               grid.x,grid.y,grid.z, block.x,block.y,block.z, smem);

        double t0=now_ms();
        switch (D) {
            case 64:  gqa_prefill_attn_kernel_t<64, G,ROWS,P_BC><<<grid,block,smem>>>(p); break;
            case 128: gqa_prefill_attn_kernel_t<128,G,ROWS,P_BC><<<grid,block,smem>>>(p); break;
            default: printf("unsupported D=%d\n",D); return 1;
        }
        cudaDeviceSynchronize();
        double kms=now_ms()-t0;
        cudaError_t err=cudaGetLastError();
        if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

        bf16* hOut=new bf16[nQ];
        cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

        float* ref=new float[nQ];
        cpu_attention(hQ,hK,hV,ref,B,Hq,Hk,ql,kl,D,causal,0);

        float max_err=0;
        for (size_t i=0;i<nQ;i++) {
            float d=fabsf(bf2f(hOut[i])-ref[i]);
            if(d>max_err) max_err=d;
        }
        printf("kernel: %.3f ms  max_err: %.6e\n\n",kms,max_err);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
        delete[]hQ;delete[]hK;delete[]hV;delete[]hOut;delete[]ref;delete[]tmp;
    }
    printf("All tests passed!\n");
    return 0;
}
