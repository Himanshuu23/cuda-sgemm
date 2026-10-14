// smem_tiled.cu - Kernel 3: shared-memory tiling (32x32 tiles).
//
// Launch: block = 32x32 threads, grid = (N/32, N/32).
//
// A block computes a 32x32 tile of C. It walks along K in steps of 32:
//   1. cooperatively load a 32x32 tile of A and a 32x32 tile of B into shared
//      memory (each thread loads exactly one element of each),
//   2. __syncthreads(),
//   3. every thread does 32 FMAs reading ONLY shared memory,
//   4. __syncthreads() again before the tiles are overwritten.
// Each value fetched from global memory is now used 32 times instead of once,
// so global traffic drops ~32x.
//
// Bank-conflict check (32 banks, 4 bytes each, warp = fixed ty, tx = 0..31):
//   As[ty][k] : same address for all lanes -> broadcast, no conflict.
//   Bs[k][tx] : 32 consecutive floats      -> one per bank, no conflict.
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
