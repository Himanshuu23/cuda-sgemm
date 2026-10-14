// naive.cu - Kernel 1: one thread per output element, "obvious" index mapping.
//
//   C[row][col] = sum_k A[row][k] * B[k][col]
//
// Launch: block = 32x32 threads, grid = (N/32, N/32).
//
// Mapping used here (deliberately the "natural but bad" one):
//   row = blockIdx.x*32 + threadIdx.x
//   col = blockIdx.y*32 + threadIdx.y
//
// A warp is 32 threads with consecutive threadIdx.x (and the same threadIdx.y).
// So one warp works on 32 DIFFERENT rows but the SAME column. Consequences:
//   * A[row*N + k]  : the 32 lanes read addresses N floats apart -> 32 separate
//                     memory sectors per load instruction (uncoalesced).
//   * B[k*N + col]  : all lanes read the same address (broadcast, cheap).
//   * C[row*N + col]: the 32 lanes write N floats apart (uncoalesced store).
// Every thread also re-reads a full row of A and column of B from global
// memory: zero data reuse inside the block beyond what the caches catch.
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
