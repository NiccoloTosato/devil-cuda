#include <cstddef>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <memory>
#include "utils.hpp"

template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        cudaFree(ptr);
    }
};

__global__ void expGPU(float *A, float *B,float *C,std::size_t elem_count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<elem_count)
      C[x]=exp(-A[x]-B[x]);
}

__global__ void process2D(float *k, float* Y,float* w_q,float* mu_g, int genes, int cells) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < genes && j < cells) {
      const int ij = i * genes + j;
      mu_g[ij] =(k[i]+Y[ij])/(1+k[i]*w_q[ij]);
    }
}

__global__ void elementWise(float *mu_g, float *w_g, std::size_t elem_count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<elem_count)
      mu_g[x]=mu_g[x]*w_g[x];
}




int main() {
  float* tmp=nullptr;  
  
  std::size_t genes{32};
  std::size_t cells{1024};
  std::size_t features{2};
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, features*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> X{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, cells*genes*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> Y{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> offset{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*features*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> mu_beta{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> cg_tmp{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> w_q{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> mu_g{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> k{tmp};

  tmp=nullptr;
  cublasHandle_t cublasH;
  CUBLAS_CHECK(cublasCreate(&cublasH));
  
  std::size_t iter{0};
  while (iter < 1) {
    ++iter;
    const float alpha{1.0f};
    const float beta{0.0f};

    // 1.0) sgemm
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, genes, cells, features, &alpha, mu_beta.get(), genes, X.get(), features, &beta, cg_tmp.get(), genes);

    // 1.1) exponential kernel
    //      C[x]=exp(-A[x]-B[x]);
    dim3 threads1D(256);
    dim3 blocks1D((genes*cells + threads1D.x - 1) / threads1D.x);
    expGPU<<<blocks1D,threads1D>>>(cg_tmp.get(),offset.get(),w_q.get(),genes*cells);

    // 2.0) 2d kernel to broadcast
    dim3 threads2D(16,16);
    dim3 blocks2D((genes + threads2D.x - 1) / threads2D.x, 
               (cells + threads2D.y - 1) / threads2D.y);
    process2D<<<blocks2D,threads2D>>>(k.get(),Y.get(),w_q.get(),mu_g.get(),genes,cells) ;

    // 3.0) elementwise multiplication
    elementWise<<<blocks1D,threads1D>>>(mu_g.get(), w_q.get(), genes*cells);

    /*
      A=torch.einsum('fc,gc->gfc',X.t(), wq_mug);
      B=torch.einsum('gfc,ck->gfk', A , X) * k.unsqueeze(1);

      Zigma = torch.inverse(B); // ma e' l'inversa calcolata piu volte ? 

      C=torch.einsum("fc,gc->gf", X.t(), (wq_mug - 1));
      
      delta = torch.einsum('gfk,gk->gf', Zigma, k * C)
      init_beta += delta //easy
      converged = torch.max(abs(delta)) < eps //usa la norma
     */
  }
}


