// kernel_common.h - include shim for the kernel files.
//
// Normal build : pulls in the real CUDA runtime header.
// CPU emulation: tests/cpu_emulation_test.cpp defines SGEMM_CPU_EMULATION and
//                SGEMM_NO_LAUNCHER, which swaps CUDA keywords for a tiny
//                thread-based emulator so the *index math* of every kernel can
//                be unit-tested on a machine without a GPU.
#pragma once
#ifdef SGEMM_CPU_EMULATION
  #include "../../tests/cuda_emul.h"
#else
  #include <cuda_runtime.h>
#endif
