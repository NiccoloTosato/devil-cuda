#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
