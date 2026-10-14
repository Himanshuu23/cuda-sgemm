#pragma once

using SgemmFn = void (*)(int N, const float* A, const float* B, float* C);

void run_naive      (int N, const float* A, const float* B, float* C);
void run_coalesced  (int N, const float* A, const float* B, float* C);
void run_smem_tiled (int N, const float* A, const float* B, float* C);
void run_reg_blocked(int N, const float* A, const float* B, float* C);
void run_vectorized (int N, const float* A, const float* B, float* C);
