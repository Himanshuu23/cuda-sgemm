#include "kernel_common.h"
#include "../sgemm.h"

constexpr int TILE = 32;

__global__ void sgemm_smem_tiled(int N, const float* __restrict__ A,
                                 const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;   // column inside the tile (lane id)
    const int ty = threadIdx.y;   // row inside the tile
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        As[ty][tx] = A[row * N + (t + tx)];        // coalesced: tx is the fast index
        Bs[ty][tx] = B[(t + ty) * N + col];        // coalesced
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

inline dim3 smem_tiled_grid(int N) { return dim3(N / TILE, N / TILE); }
inline dim3 smem_tiled_block()     { return dim3(TILE, TILE); }

// ---- host launcher (excluded by the CPU emulation test) ----
#ifndef SGEMM_NO_LAUNCHER
void run_smem_tiled(int N, const float* A, const float* B, float* C) {
    sgemm_smem_tiled<<<smem_tiled_grid(N), smem_tiled_block()>>>(N, A, B, C);
}
#endif
