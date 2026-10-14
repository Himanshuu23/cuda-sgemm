// cuda_emul.h - a minimal CPU emulator for the CUDA constructs used by the kernels.
//
// Purpose: validate thread/block index math and shared-memory tiling logic
// without a GPU. It is NOT a performance tool and supports only what these
// kernels use: __global__, __shared__, __syncthreads, threadIdx/blockIdx/
// blockDim/gridDim, dim3, float4.
//
// How it works: blocks run one after another; each thread of a block is a real
// std::thread, and __syncthreads() is a std::barrier. A function-local `static`
// array stands in for __shared__ memory (all threads of the running block see
// the same one, and only one block runs at a time).
#pragma once
#include <barrier>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define __syncthreads() (g_emul_barrier->arrive_and_wait())

struct float4 { float x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

inline dim3 blockIdx, blockDim, gridDim;           // same for all threads of a running block
inline thread_local dim3 threadIdx;                // per-thread
inline std::barrier<>* g_emul_barrier = nullptr;

template <class Kernel, class... Args>
void emu_launch(dim3 grid, dim3 block, Kernel kernel, Args... args) {
    gridDim = grid;
    blockDim = block;
    const unsigned nthreads = block.x * block.y * block.z;
    for (unsigned by = 0; by < grid.y; ++by) {
        for (unsigned bx = 0; bx < grid.x; ++bx) {
            blockIdx = dim3(bx, by, 0);
            std::barrier<> bar(nthreads);
            g_emul_barrier = &bar;
            std::vector<std::thread> threads;
            threads.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                threads.emplace_back([=, &block] {
                    threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
                    kernel(args...);
                });
            }
            for (auto& th : threads) th.join();
            g_emul_barrier = nullptr;
        }
    }
}
