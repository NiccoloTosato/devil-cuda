#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverDn.h>
#include "utils.hpp"
#include <cassert>
#include <iostream>
#include <ostream>

__global__ void initIdentityGPU(float *Matrix, int rows, int cols,float alpha) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if(y < rows && x < cols) {
    if(x == y)
      Matrix[y*cols+x] = 1*alpha;
    else
      Matrix[y*cols+x] = 0;
  }
}

int inverseMatrix(cusolverDnHandle_t cusolverH, float *A_device, float *A_inv_device, int n){

  // preliminaries declarations
  int *pivot = NULL; // pivot indices
  int *info = NULL; // error info
  int lwork = 0; // size of workspace
  float *workspace = NULL; // device workspace for getrf

  //allocation for pivot and info
  CUDA_CHECK(cudaMallocManaged(&pivot, sizeof(int)*n));
  CUDA_CHECK(cudaMallocManaged(&info, sizeof(int)));

  //create solver,and get buffersize
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolverH, n, n, A_device, n, &lwork));
  
  //allocation for workspace
  CUDA_CHECK(cudaMallocManaged(&workspace, sizeof(float) * lwork));

  //factorize ! 
  CUSOLVER_CHECK(cusolverDnSgetrf(cusolverH, n, n, A_device, n, workspace, pivot, info));
  if (*info != 0 ) {
    CUDA_CHECK(cudaDeviceSynchronize());
    assert(0 == info[0]);
  }

  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
		     (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
  initIdentityGPU<<<blocksPerGrid, threadsPerBlock>>>(A_inv_device, n, n, 1.0);
  //use this to allocate the necessary space !!!!
  //cusolverDnSgetrf(cusolverH, n, n, A_device, n, d_work, d_Ipiv, d_info);
  CUSOLVER_CHECK( cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, A_device, n, pivot,
				   A_inv_device, // This should be the identity matrix
				   n, info) );
  return 0; 
  
}
