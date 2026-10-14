#include "kernel_common.h"
#include "../sgemm.h"

constexpr int RB_BM = 128, RB_BN = 128, RB_BK = 8;
constexpr int RB_TM = 8,   RB_TN = 8;
constexpr int RB_THREADS = 256;
constexpr int RB_TX = RB_BN / RB_TN;   // 16 threads across

__global__ void __launch_bounds__(RB_THREADS)
sgemm_reg_blocked(int N, const float* __restrict__ A,
                  const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[RB_BM][RB_BK];   // 128 x 8  (row-major: As[m][k])
    __shared__ float Bs[RB_BK][RB_BN];   // 8 x 128  (row-major: Bs[k][n])

    const int tid = threadIdx.x;
    const int tx = tid % RB_TX;          // 0..15
    const int ty = tid / RB_TX;          // 0..15

    // Top-left corners of this block's slices of A (a 128-row strip) and B (a 128-col strip).
    const float* A_blk = A + (size_t)blockIdx.y * RB_BM * N;
    const float* B_blk = B + (size_t)blockIdx.x * RB_BN;

    float acc[RB_TM][RB_TN] = {};        // 64 accumulators, live in registers
    float ra[RB_TM], rb[RB_TN];          // operand fragments

    for (int k0 = 0; k0 < N; k0 += RB_BK) {
        // Cooperative load: 1024 elements per tile / 256 threads = 4 each.
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int e = tid + r * RB_THREADS;      // 0..1023
            const int a_m = e / RB_BK, a_k = e % RB_BK;   // 8 consecutive threads = one row of 8 floats
            As[a_m][a_k] = A_blk[(size_t)a_m * N + k0 + a_k];
            const int b_k = e / RB_BN, b_n = e % RB_BN;   // 128 consecutive threads = one row of 128 floats
            Bs[b_k][b_n] = B_blk[(size_t)(k0 + b_k) * N + b_n];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < RB_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < RB_TM; ++i) ra[i] = As[i * 16 + ty][k];
            #pragma unroll
            for (int j = 0; j < RB_TN; ++j) rb[j] = Bs[k][j * 16 + tx];
            #pragma unroll
            for (int i = 0; i < RB_TM; ++i)
                #pragma unroll
                for (int j = 0; j < RB_TN; ++j)
                    acc[i][j] += ra[i] * rb[j];      // outer product, 64 FMAs
        }
        __syncthreads();
    }

    // Write back. For fixed i, a warp stores 16 consecutive floats per ty value.
    #pragma unroll
    for (int i = 0; i < RB_TM; ++i) {
        const int row = blockIdx.y * RB_BM + i * 16 + ty;
        #pragma unroll
        for (int j = 0; j < RB_TN; ++j) {
            const int col = blockIdx.x * RB_BN + j * 16 + tx;
            C[(size_t)row * N + col] = acc[i][j];
        }
    }
}

inline dim3 reg_blocked_grid(int N) { return dim3(N / RB_BN, N / RB_BM); }
inline dim3 reg_blocked_block()     { return dim3(RB_THREADS); }

// ---- host launcher (excluded by the CPU emulation test) ----
#ifndef SGEMM_NO_LAUNCHER
void run_reg_blocked(int N, const float* A, const float* B, float* C) {
    sgemm_reg_blocked<<<reg_blocked_grid(N), reg_blocked_block()>>>(N, A, B, C);
}
#endif
