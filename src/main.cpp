#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "einsum.hpp"
#include "kernel.h"
#include "inverse.hpp"
#include "utils.hpp"
#include "cutensor.h"

template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        cudaFree(ptr);
    }
};

std::vector<float> readDatFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Unable to open file " << filename << std::endl;
    return {};
  }
  // Get the file size
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  // Read the data
  std::vector<float> data(size / sizeof(float));
  if (file.read(reinterpret_cast<char *>(data.data()), size)) {
    std::cout << "Loading file " << filename << " Success,size: " << data.size() << std::endl;
    return data;
  } else {
    std::cerr << "Error reading file " << filename << std::endl;
    return {};
  }
}

__global__ void printMatrix(const int rows,const int cols, float* const matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
	  printf("%2.4f ", matrix[i*cols+j]);
        }
        printf("\n");
    }
}

__global__ void printMatrixT(const int rows,const int cols, float* const matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
	  printf("%2.3f ", matrix[j*rows+i]);
        }
        printf("\n");
    }
}

void toGPU(const std::vector<float> &vec,float* const vec_gpu) {
  CUDA_CHECK( cudaMemcpy(vec_gpu, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice) );
}

int main(int argc,char* argv[]) {
  float* tmp=nullptr;
  /******************************
   * Shape definition 
   ******************************/
  const std::size_t genes{64};
  const std::size_t cells{1024};
  const std::size_t features{2};

  /* small debug files in ../data-debug/
  std::size_t genes{3};
  std::size_t cells{4};
  std::size_t features{2};
  */

  /******************************
   * Allocate GPU memory, read data and move to GPU. 
   ******************************/
  CUDA_CHECK( cudaMalloc((void**)&tmp, features*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> X{tmp};
  const auto X_host = readDatFile("../data/X.dat");
  toGPU(X_host, X.get());
  std::cout << "X {"<<cells<<","<<features <<"}\n";
  std::cout << std::flush;
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, cells*genes*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> Y{tmp};
  const auto Y_host = readDatFile("../data/Y.dat");
  toGPU(Y_host, Y.get());
  std::cout << "Y {"<<genes<<","<<cells<<"}\n";
  std::cout << std::flush;
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> offset{tmp};
  const auto offset_host = readDatFile("../data/off.dat");
  toGPU(offset_host, offset.get());
  std::cout << "offset {"<<genes<<","<<cells <<"}\n";
  std::cout << std::flush;

  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*features*sizeof(float)) );
  std::unique_ptr<float, CudaDeleter<float>> mu_beta{tmp};
  const auto mu_beta_host = readDatFile("../data/mu_beta.dat");
  toGPU(mu_beta_host, mu_beta.get());
  std::cout << "mu_beta {"<<genes<<","<<features <<"}\n";
  std::cout << std::flush;

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> k{tmp};
  auto k_host = readDatFile("../data/K.dat");
  for (auto &x : k_host)
    x=1/x;
  toGPU(k_host, k.get());
  std::cout << "K {"<<genes<<","<<1 <<"}\n";
  std::cout << std::flush;
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float, CudaDeleter<float>> cg_tmp{tmp};
  cudaMemset(cg_tmp.get(), 0, genes*cells*sizeof(float));
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> w_q{tmp};
  cudaMemset(w_q.get(), 0, genes * cells * sizeof(float));
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> mu_g{tmp};
  cudaMemset(mu_g.get(), 0, genes*cells*sizeof(float));
  tmp = nullptr;

  /******************************
   * Create handlers and setup
   ******************************/
  cublasHandle_t cublasH;
  CUBLAS_CHECK( cublasCreate(&cublasH) );
  cutensorHandle_t cutensorH;
  CUTENSOR_CHECK( cutensorCreate(&cutensorH) );
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK( cutensorHandleResizePlanCache(cutensorH, numCachelines) );
  //cusolverDnHandle_t cusolverH;
  //CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );
  
  /******************************
   * Initialize tensor product
   ******************************/
  EinsumWrapper einsum_offsetT { std::string{"ij->ji"},
				 {(int)genes, (int)cells},
				 {}};
  EinsumWrapper einsum_cg_tmp2 {std::string{"ik,jk->ij"},
				{(int)cells, (int)features},
				{(int)genes, (int)features}};
  EinsumWrapper einsum_w_qT { std::string{"ij->ji"},
			      {(int)cells, (int)genes},
			      {}};
  EinsumWrapper einsum_A{ std::string{"cf,gc->gfc"},
                         {(int)cells, (int)features},
                         {(int)genes, (int)cells}};
  EinsumWrapper einsum_B{ std::string{"gfc,ck->gfk"},
			  {(int)genes, (int)features, (int)cells},
			  {(int)cells, (int)features}};
  EinsumWrapper einsum_Bk{ std::string{"gfc,g->gfc"},
                          {(int)genes, (int)features, (int)features},
                          {(int)genes}};
  EinsumWrapper einsum_C{ std::string{"cf,gc->gf"},
			  {(int)cells, (int)features},
			  {(int)genes, (int)cells}};
  EinsumWrapper einsum_last { std::string{"gk,gf->gf"},
			      {(int)genes, 1},
			      {(int)genes, (int)features}};
  EinsumWrapper einsum_delta{ std::string{"gfk,gk->gf"},
			      {(int)genes, (int)features, (int)features},
			      {(int)genes, (int)features}};

  /******************************
   * This allocate the workspace and the output tensor
   ******************************/
  float *w_qT=einsum_w_qT.allocate();
  float *offsetT = einsum_offsetT.allocate();
  float *cg_tmp2 = einsum_cg_tmp2.allocate();
  float *A = einsum_A.allocate();
  float *B = einsum_B.allocate();
  float *C = einsum_C.allocate();
  float* Bk = einsum_Bk.allocate();
  float *delta = einsum_delta.allocate();
  float* last=einsum_last.allocate();
  einsum_offsetT.execute(cutensorH, offset.get(), nullptr);
  //free offset memory after transposition
  offset.reset();

  /******************************
   * Allocate Zigma, The array of pointer to Zigma and Bk
   ******************************/
  float **Zigma_pointer;
  float **Bk_pointer; 
  float *Zigma;
  //Use Managed memory to simply set the addresses
  CUDA_CHECK(cudaMallocManaged(&Zigma_pointer, genes * sizeof(float*)) );
  CUDA_CHECK(cudaMallocManaged(&Bk_pointer, genes * sizeof(float*)) );
  CUDA_CHECK(cudaMalloc(&Zigma, sizeof(float) * features * features * genes));
  for (int i = 0; i < genes; ++i) {
    Zigma_pointer[i] = Zigma + features * features * i;
    Bk_pointer[i] = Bk + features * features * i;
  }

  /******************************
   * Initialize norm s.t. the initial check is always True , set iter to 0,
   *measure start time.
   ******************************/
  //I know, there is a narrow conversion here.
  float norm(std::sqrt(genes*features));
  std::size_t iter{0};
  auto t1 = std::chrono::high_resolution_clock::now();

  while (iter < 8000 && (norm/std::sqrt(genes*features)) > 1E-12) {
    ++iter;
    einsum_cg_tmp2.execute(cutensorH, X.get(), mu_beta.get());
    dim3 threads1D(256);
    dim3 blocks1D((genes * cells + threads1D.x - 1) / threads1D.x);
    expGPU<<<blocks1D, threads1D>>>(cg_tmp2, offsetT, w_q.get(),
                                    genes * cells);
    einsum_w_qT.execute(cutensorH,w_q.get(),nullptr);
    dim3 threads2D(16,16);
    dim3 blocks2D((cells + threads2D.x - 1) / threads2D.x,
                  (genes + threads2D.y - 1) / threads2D.y);
    process2D<<<blocks2D, threads2D>>>(k.get(), Y.get(), w_qT, mu_g.get(),
                                       genes, cells);
    elementWise<<<blocks1D, threads1D>>>(mu_g.get(), w_qT, genes * cells);
    einsum_A.execute(cutensorH, X.get(), mu_g.get());
    einsum_B.execute(cutensorH, A, X.get());
    einsum_Bk.execute(cutensorH,B,k.get());
    inverseMatrix2(cublasH, Bk_pointer, Zigma_pointer, features ,genes);
    elementWiseSub<<<blocks1D,threads1D>>>(mu_g.get(), genes*cells);
    einsum_C.execute(cutensorH,X.get(),mu_g.get()); 
    einsum_last.execute(cutensorH,k.get(),C);
    einsum_delta.execute(cutensorH,Zigma,last);
    final1D<<<blocks1D,threads1D>>>(mu_beta.get(),delta,genes*features);
    cublasSnrm2(cublasH, genes * features, delta, 1, &norm);
    //std::cout << "Norm " << norm/std::sqrt(genes*features)<< std::endl;
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaFree(Zigma));
  CUDA_CHECK(cudaFree(Bk_pointer));
  CUDA_CHECK(cudaFree(Zigma_pointer));
  CUDA_CHECK(cudaFree(w_qT));
  CUDA_CHECK(cudaFree(offsetT));
  CUDA_CHECK(cudaFree(cg_tmp2));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(Bk));
  CUDA_CHECK(cudaFree(delta));
  CUDA_CHECK(cudaFree(last));

  /*********************
   * Destroy handles
   ********************/
  CUBLAS_CHECK( cublasDestroy(cublasH) );
  CUTENSOR_CHECK( cutensorDestroy(cutensorH) );

  std::cout << "mu_beta {"<<genes<<","<<features <<"}\n";
  printMatrix<<<1, 1>>>( genes,features, mu_beta.get());
  cudaDeviceSynchronize();
  std::cout << std::flush;
  std::cout << "Norm " << norm / std::sqrt(genes * features) << std::endl;

  auto elapsed{t2-t1};
  std::cout<<std::chrono::duration<double,std::milli>(elapsed).count() / iter << " ms [avg iter time]"<<std::endl;

}


