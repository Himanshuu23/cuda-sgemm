// cpu_emulation_test.cpp - checks the index math of all five kernels on the CPU.
//
// The kernel .cu files are #included as plain C++ with CUDA keywords mapped to
// a thread-based emulator (tests/cuda_emul.h). Each emulated kernel is compared
// against a double-precision triple loop. This validates tiling, thread->element
// mapping, shared-memory indexing and the float4/transpose logic. It does NOT
// say anything about performance, bank conflicts or real-GPU behaviour.
//
// Build & run:  make emu-test
#define SGEMM_CPU_EMULATION
#define SGEMM_NO_LAUNCHER
#include "cuda_emul.h"
#include "../src/kernels/naive.cu"
#include "../src/kernels/coalesced.cu"
#include "../src/kernels/smem_tiled.cu"
#include "../src/kernels/reg_blocked.cu"
#include "../src/kernels/vectorized.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct Case {
    const char* name;
    void (*launch)(int, const float*, const float*, float*);
};

#define MAKE_CASE(NAME, KERNEL, GRID, BLOCK)                                       \
    {NAME, [](int n, const float* a, const float* b, float* c) {                   \
         emu_launch(GRID(n), BLOCK(), KERNEL, n, a, b, c);                         \
     }}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 256;   // multiple of 128, >1 tile in each dim
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> A(N * N), B(N * N), C(N * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    std::vector<double> ref((size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0;
            for (int k = 0; k < N; ++k) s += (double)A[i * N + k] * B[k * N + j];
            ref[(size_t)i * N + j] = s;
        }

    const Case cases[] = {
        MAKE_CASE("naive",       sgemm_naive,       naive_grid,       naive_block),
        MAKE_CASE("coalesced",   sgemm_coalesced,   coalesced_grid,   coalesced_block),
        MAKE_CASE("smem_tiled",  sgemm_smem_tiled,  smem_tiled_grid,  smem_tiled_block),
        MAKE_CASE("reg_blocked", sgemm_reg_blocked, reg_blocked_grid, reg_blocked_block),
        MAKE_CASE("vectorized",  sgemm_vectorized,  vectorized_grid,  vectorized_block),
    };

    int failures = 0;
    for (const Case& c : cases) {
        std::fill(C.begin(), C.end(), std::nanf(""));   // NaN: catches unwritten outputs
        c.launch(N, A.data(), B.data(), C.data());
        double max_abs = 0;
        bool ok = true;
        for (size_t i = 0; i < (size_t)N * N; ++i) {
            double d = std::fabs((double)C[i] - ref[i]);
            if (!(d <= 1e-3)) ok = false;               // also true for NaN
            if (d > max_abs) max_abs = d;
        }
        std::printf("%-12s N=%d  max_abs_err=%.3e  %s\n", c.name, N, max_abs, ok ? "PASS" : "FAIL");
        failures += !ok;
    }
    return failures ? 1 : 0;
}
