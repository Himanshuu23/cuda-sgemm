#include "kernel_common.h"
#include "../sgemm.h"

constexpr int NAIVE_BS = 32;

__global__ void sgemm_naive(int N, const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C) {
    const int row = blockIdx.x * NAIVE_BS + threadIdx.x;  // fast index -> row (bad for coalescing)
    const int col = blockIdx.y * NAIVE_BS + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

inline dim3 naive_grid(int N) { return dim3(N / NAIVE_BS, N / NAIVE_BS); }
inline dim3 naive_block()     { return dim3(NAIVE_BS, NAIVE_BS); }

// ---- host launcher (excluded by the CPU emulation test) ----
#ifndef SGEMM_NO_LAUNCHER
void run_naive(int N, const float* A, const float* B, float* C) {
    sgemm_naive<<<naive_grid(N), naive_block()>>>(N, A, B, C);
}
#endif
