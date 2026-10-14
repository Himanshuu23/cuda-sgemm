#pragma once
#ifdef SGEMM_CPU_EMULATION
  #include "../../tests/cuda_emul.h"
#else
  #include <cuda_runtime.h>
#endif
