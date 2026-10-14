#include "kernel_common.h"
#include "../sgemm.h"

constexpr int V_BM = 128, V_BN = 128, V_BK = 8;
constexpr int V_THREADS = 256;
constexpr int V_TX = 16;                 // threads across (V_BN / 8)
constexpr int V_A_LD = V_BM + 4;         // padded leading dimension of the transposed A tile

__global__ void __launch_bounds__(V_THREADS)
sgemm_vectorized(int N, const float* __restrict__ A,
                 const float* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(16) float As[V_BK * V_A_LD];   // As[k][m], padded rows
    __shared__ __align__(16) float Bs[V_BK * V_BN];     // Bs[k][n]

    const int tid = threadIdx.x;
    const int tx = tid % V_TX;           // 0..15  -> column group
    const int ty = tid / V_TX;           // 0..15  -> row group

    const float* A_blk = A + (size_t)blockIdx.y * V_BM * N;
    const float* B_blk = B + (size_t)blockIdx.x * V_BN;

    // Load assignment: one float4 of A and one float4 of B per thread per K step.
    const int a_m = tid / 2;             // 0..127 : row of the A tile
    const int a_k = (tid % 2) * 4;       // 0 or 4 : first of 4 consecutive k
    const int b_k = tid / 32;            // 0..7   : row of the B tile
    const int b_n = (tid % 32) * 4;      // 0..124 : first of 4 consecutive columns

    float acc[8][8] = {};
    float ra[8], rb[8];

    for (int k0 = 0; k0 < N; k0 += V_BK) {
        // ---- global -> shared (A transposed on the way in) ----
        const float4 av = *reinterpret_cast<const float4*>(A_blk + (size_t)a_m * N + k0 + a_k);
        As[(a_k + 0) * V_A_LD + a_m] = av.x;
        As[(a_k + 1) * V_A_LD + a_m] = av.y;
        As[(a_k + 2) * V_A_LD + a_m] = av.z;
        As[(a_k + 3) * V_A_LD + a_m] = av.w;

        const float4 bv = *reinterpret_cast<const float4*>(B_blk + (size_t)(k0 + b_k) * N + b_n);
        *reinterpret_cast<float4*>(&Bs[b_k * V_BN + b_n]) = bv;
        __syncthreads();

        // ---- compute: 8 K steps of an 8x8 outer product ----
        #pragma unroll
        for (int k = 0; k < V_BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[k * V_A_LD + ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[k * V_A_LD + 64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k * V_BN + tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k * V_BN + 64 + tx * 4]);
            ra[0] = a0.x; ra[1] = a0.y; ra[2] = a0.z; ra[3] = a0.w;
            ra[4] = a1.x; ra[5] = a1.y; ra[6] = a1.z; ra[7] = a1.w;
            rb[0] = b0.x; rb[1] = b0.y; rb[2] = b0.z; rb[3] = b0.w;
            rb[4] = b1.x; rb[5] = b1.y; rb[6] = b1.z; rb[7] = b1.w;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    // ---- write back with float4 stores ----
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row_in_tile = (i < 4) ? (ty * 4 + i) : (64 + ty * 4 + (i - 4));
        float* c_row = C + (size_t)(blockIdx.y * V_BM + row_in_tile) * N + blockIdx.x * V_BN;
        *reinterpret_cast<float4*>(c_row + tx * 4) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(c_row + 64 + tx * 4) =
            make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

inline dim3 vectorized_grid(int N) { return dim3(N / V_BN, N / V_BM); }
inline dim3 vectorized_block()     { return dim3(V_THREADS); }

// ---- host launcher (excluded by the CPU emulation test) ----
#ifndef SGEMM_NO_LAUNCHER
void run_vectorized(int N, const float* A, const float* B, float* C) {
    sgemm_vectorized<<<vectorized_grid(N), vectorized_block()>>>(N, A, B, C);
}
#endif
