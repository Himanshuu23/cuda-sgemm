// sgemm.h - the single host-side signature shared by every kernel in the ladder.
//
// Every run_* function computes   C = A * B   (alpha = 1, beta = 0)
// for square, row-major, fp32 N x N matrices that already live in device memory.
// The launchers are asynchronous (they only enqueue the kernel on the default
// stream); the harness is responsible for synchronising and error checking.
//
// Requirement for ALL kernels: N must be a multiple of 128 (the largest tile).
// That keeps the kernels free of bounds checks so the inner loops stay clean.
#pragma once

using SgemmFn = void (*)(int N, const float* A, const float* B, float* C);

void run_naive      (int N, const float* A, const float* B, float* C);
void run_coalesced  (int N, const float* A, const float* B, float* C);
void run_smem_tiled (int N, const float* A, const float* B, float* C);
void run_reg_blocked(int N, const float* A, const float* B, float* C);
void run_vectorized (int N, const float* A, const float* B, float* C);
