// main.cu - benchmark + verification harness for the SGEMM optimisation ladder.
//
// For every requested size N it:
//   1. creates random N x N matrices A and B (fixed seed, values in [-1, 1]),
//   2. computes the reference C with cublasSgemm,
//   3. runs each of our kernels, compares against the reference
//      (max abs error, error relative to max|C_ref|) and prints PASS/FAIL,
//   4. times each kernel and cuBLAS with cudaEvents: warmup runs, then the
//      MEDIAN of `iters` timed runs (one launch per event pair),
//   5. prints GFLOPS and % of cuBLAS, and appends everything to a CSV.
//
// Usage:
//   sgemm_bench [--kernel all|naive|coalesced|smem_tiled|reg_blocked|vectorized]
//               [--sizes 512,1024,2048,4096] [--warmup 3] [--iters 20]
//               [--csv results/results.csv] [--profile]
//
// --profile : run exactly ONE launch of ONE kernel at the first size, with no
//             cuBLAS and no verification. Used by scripts/profile.sh so that
//             Nsight Compute sees a single, clean kernel launch.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "sgemm.h"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err_), #call);                            \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t st_ = (call);                                                  \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                           \
            std::fprintf(stderr, "cuBLAS error %s:%d: status %d (%s)\n", __FILE__,    \
                         __LINE__, (int)st_, #call);                                  \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

struct KernelEntry {
    const char* name;
    SgemmFn fn;
};

static const KernelEntry kKernels[] = {
    {"naive", run_naive},
    {"coalesced", run_coalesced},
    {"smem_tiled", run_smem_tiled},
    {"reg_blocked", run_reg_blocked},
    {"vectorized", run_vectorized},
};

// ---------------------------------------------------------------- helpers ----

static std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        out.push_back(std::atoi(s.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    return out;
}

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// cuBLAS is column-major. Row-major C = A*B is the same memory as column-major
// C^T = B^T * A^T, so we call sgemm with the operands swapped (B first, then A).
static void cublas_sgemm_rowmajor(cublasHandle_t h, int N, const float* A, const float* B, float* C) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N));
}

// Time `launch()` : `warmup` untimed runs, then `iters` runs each bracketed by
// its own event pair. Returns the median time in milliseconds.
template <class Launch>
static double time_median_ms(Launch launch, int warmup, int iters) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return median(ms);
}

static double gflops(int N, double ms) { return 2.0 * N * (double)N * N / (ms * 1e6); }

struct Row {
    std::string kernel;
    int N;
    double ms, gflops, pct_cublas, max_abs, rel_err;
    const char* status;
};

// ------------------------------------------------------------------- main ----

int main(int argc, char** argv) {
    std::string kernel_sel = "all";
    std::string sizes_arg = "512,1024,2048,4096";
    std::string csv_path = "results/results.csv";
    int warmup = 3, iters = 20;
    bool profile = false;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--kernel")) kernel_sel = need("--kernel");
        else if (!std::strcmp(argv[i], "--sizes")) sizes_arg = need("--sizes");
        else if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(need("--warmup"));
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(need("--iters"));
        else if (!std::strcmp(argv[i], "--csv")) csv_path = need("--csv");
        else if (!std::strcmp(argv[i], "--profile")) profile = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    if (iters < 1) { std::fprintf(stderr, "--iters must be >= 1\n"); return 2; }

    // Select kernels.
    std::vector<KernelEntry> selected;
    for (const auto& k : kKernels)
        if (kernel_sel == "all" || kernel_sel == k.name) selected.push_back(k);
    if (selected.empty()) { std::fprintf(stderr, "unknown kernel '%s'\n", kernel_sel.c_str()); return 2; }
    const std::vector<int> sizes = parse_sizes(sizes_arg);
    for (int n : sizes)
        if (n <= 0 || n % 128 != 0) { std::fprintf(stderr, "N=%d must be a positive multiple of 128\n", n); return 2; }

    // Device info.
    int ndev = 0;
    cudaError_t derr = cudaGetDeviceCount(&ndev);
    if (derr != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "No usable CUDA device found (%s).\n", cudaGetErrorString(derr));
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
#if CUDART_VERSION < 13000
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
#endif
    std::printf("GPU: %s | cc %d.%d | %d SMs | %.2f GB\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    if (prop.major == 8 && prop.minor == 6 && clock_khz > 0) {
        // cc 8.6: 128 FP32 lanes per SM, 1 FMA = 2 FLOP per lane per cycle.
        std::printf("Theoretical FP32 peak at max reported clock (%.0f MHz): %.0f GFLOPS "
                    "(real sustained clock is usually lower)\n",
                    clock_khz / 1000.0, prop.multiProcessorCount * 128.0 * 2.0 * clock_khz / 1e6);
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));  // strict FP32 (no TF32 tensor cores)

    // ------------------------------------------------------- profile mode ----
    if (profile) {
        const int N = sizes[0];
        const size_t bytes = (size_t)N * N * sizeof(float);
        std::vector<float> h((size_t)N * N);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes)); CUDA_CHECK(cudaMalloc(&dC, bytes));
        for (auto& v : h) v = dist(rng);
        CUDA_CHECK(cudaMemcpy(dA, h.data(), bytes, cudaMemcpyHostToDevice));
        for (auto& v : h) v = dist(rng);
        CUDA_CHECK(cudaMemcpy(dB, h.data(), bytes, cudaMemcpyHostToDevice));
        selected[0].fn(N, dA, dB, dC);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("profile run done: %s N=%d\n", selected[0].name, N);
        return 0;
    }

    // ---------------------------------------------------------- benchmark ----
    std::vector<Row> rows;
    for (int N : sizes) {
        const size_t count = (size_t)N * N, bytes = count * sizeof(float);
        std::vector<float> hA(count), hB(count), hC(count), hRef(count);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& v : hA) v = dist(rng);
        for (auto& v : hB) v = dist(rng);

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes)); CUDA_CHECK(cudaMalloc(&dRef, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        // Reference + cuBLAS timing.
        cublas_sgemm_rowmajor(handle, N, dA, dB, dRef);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
        double ref_max = 0;
        for (float v : hRef) ref_max = std::max(ref_max, (double)std::fabs(v));

        const double cublas_ms = time_median_ms([&] { cublas_sgemm_rowmajor(handle, N, dA, dB, dC); }, warmup, iters);
        const double cublas_gf = gflops(N, cublas_ms);
        rows.push_back({"cublas", N, cublas_ms, cublas_gf, 100.0, 0.0, 0.0, "REF"});

        for (const auto& k : selected) {
            // Fill C with NaN bytes so an element the kernel forgets to write fails verification.
            CUDA_CHECK(cudaMemset(dC, 0xFF, bytes));
            k.fn(N, dA, dB, dC);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

            double max_abs = 0;
            bool finite = true;
            for (size_t i = 0; i < count; ++i) {
                const double d = std::fabs((double)hC[i] - (double)hRef[i]);
                if (!(d <= 1e30)) finite = false;  // NaN / inf
                else max_abs = std::max(max_abs, d);
            }
            const double rel = max_abs / std::max(ref_max, 1e-30);
            const bool pass = finite && rel < 1e-4;  // FP32 accumulation-order noise is ~1e-6

            const double ms = time_median_ms([&] { k.fn(N, dA, dB, dC); }, warmup, iters);
            const double gf = gflops(N, ms);
            rows.push_back({k.name, N, ms, gf, 100.0 * gf / cublas_gf, max_abs, rel, pass ? "PASS" : "FAIL"});
        }

        // Print this size's table.
        std::printf("\nN = %d  (median of %d runs after %d warmup)\n", N, iters, warmup);
        std::printf("%-12s %10s %10s %10s %12s %10s  %s\n", "kernel", "ms", "GFLOPS", "%cuBLAS", "max_abs_err", "rel_err", "check");
        for (const Row& r : rows)
            if (r.N == N)
                std::printf("%-12s %10.3f %10.1f %9.1f%% %12.3e %10.2e  %s\n", r.kernel.c_str(), r.ms, r.gflops,
                            r.pct_cublas, r.max_abs, r.rel_err, r.status);
        std::fflush(stdout);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dRef));
    }

    // CSV.
    if (FILE* f = std::fopen(csv_path.c_str(), "w")) {
        std::fprintf(f, "kernel,N,median_ms,gflops,pct_of_cublas,max_abs_err,rel_err,status\n");
        for (const Row& r : rows)
            std::fprintf(f, "%s,%d,%.6f,%.3f,%.3f,%.6e,%.6e,%s\n", r.kernel.c_str(), r.N, r.ms, r.gflops,
                         r.pct_cublas, r.max_abs, r.rel_err, r.status);
        std::fclose(f);
        std::printf("\nWrote %s\n", csv_path.c_str());
    } else {
        std::fprintf(stderr, "could not open %s for writing (does the directory exist?)\n", csv_path.c_str());
    }

    int fails = 0;
    for (const Row& r : rows) fails += (std::strcmp(r.status, "FAIL") == 0);
    CUBLAS_CHECK(cublasDestroy(handle));
    return fails ? 3 : 0;
}
