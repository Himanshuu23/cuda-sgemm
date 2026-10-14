cmake_minimum_required(VERSION 3.18)
project(cuda_sgemm LANGUAGES CXX CUDA)

# RTX 2050 = Ampere GA107, compute capability 8.6. Needs CUDA >= 11.1.
# Override with -DCMAKE_CUDA_ARCHITECTURES=<n> for a different GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_bench
  src/main.cu
  src/kernels/naive.cu
  src/kernels/coalesced.cu
  src/kernels/smem_tiled.cu
  src/kernels/reg_blocked.cu
  src/kernels/vectorized.cu)

# -lineinfo lets Nsight Compute map SASS back to source lines (no runtime cost).
# We deliberately do NOT use --use_fast_math: it would change numerics vs cuBLAS.
target_compile_options(sgemm_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(sgemm_bench PRIVATE CUDA::cublas CUDA::cudart)
