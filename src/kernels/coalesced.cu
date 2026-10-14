#include "kernel_common.h"
#include "../sgemm.h"

constexpr int COAL_BS = 32;

__global__ void sgemm_coalesced(int N, const float* __restrict__ A,
                                const float* __restrict__ B, float* __restrict__ C) {
    const int row = blockIdx.y * COAL_BS + threadIdx.x / COAL_BS;
    const int col = blockIdx.x * COAL_BS + threadIdx.x % COAL_BS;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

inline dim3 coalesced_grid(int N) { return dim3(N / COAL_BS, N / COAL_BS); }
inline dim3 coalesced_block()     { return dim3(COAL_BS * COAL_BS); }

// ---- host launcher (excluded by the CPU emulation test) ----
#ifndef SGEMM_NO_LAUNCHER
void run_coalesced(int N, const float* A, const float* B, float* C) {
    sgemm_coalesced<<<coalesced_grid(N), coalesced_block()>>>(N, A, B, C);
}
#endif
