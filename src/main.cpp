#include <cstddef>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <memory>
#include <string>
#include <cusolverDn.h>
#include "cutensor.h"
#include "einsum.hpp"
#include "inverse.hpp"
#include "utils.hpp"
#include <fstream>
#include <vector>
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

__global__ void elementWiseSub(float *mu_g, std::size_t elem_count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<elem_count)
      mu_g[x]=mu_g[x]-1;
}

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


void toGPU(std::vector<float> &vec, float* vec_gpu) {
  CUDA_CHECK( cudaMemcpy(vec_gpu, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice) );
}

int main() {
  float* tmp=nullptr;  
  
  std::size_t genes{64};
  std::size_t cells{1024};
  std::size_t features{2};
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, features*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> X{tmp};
  auto X_host = readDatFile("../data/X.dat");
  toGPU(X_host, X.get());
    
  CUDA_CHECK( cudaMalloc((void**)&tmp, cells*genes*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> Y{tmp};
  auto Y_host = readDatFile("../data/Y.dat");
  toGPU(Y_host, Y.get());
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> offset{tmp};
  auto offset_host = readDatFile("../data/off.dat");
  toGPU(offset_host, offset.get());
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*features*sizeof(float)) );
  std::unique_ptr<float, CudaDeleter<float>> mu_beta{tmp};
  auto mu_beta_host = readDatFile("../data/mu_beta.dat");
  toGPU(mu_beta_host, mu_beta.get());
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> k{tmp};
  auto k_host = readDatFile("../data/K.dat");
  toGPU(k_host, k.get());
  
  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float, CudaDeleter<float>> cg_tmp{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> w_q{tmp};

  CUDA_CHECK( cudaMalloc((void**)&tmp, genes*cells*sizeof(float)) );
  std::unique_ptr<float,CudaDeleter<float>> mu_g{tmp};


  tmp = nullptr;

  /////////////////////////////////////////////////////////////
  cublasHandle_t cublasH;
  CUBLAS_CHECK(cublasCreate(&cublasH));
  cutensorHandle_t cutensorH;
  CUTENSOR_CHECK( cutensorCreate(&cutensorH) );
  /**********************
   * Setup planCache (optional)
   **********************/
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK(cutensorHandleResizePlanCache(cutensorH, numCachelines));
  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );
  //////////////////////////////////////////////////////////////


  
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
      create A
      A=torch.einsum('fc,gc->gfc',X.t(), wq_mug);
    */
    std::unique_ptr<float, CudaDeleter<float>> A{(float *)general_einsum(cutensorH, {(int)cells,(int) features}, {(int) genes, (int)cells},X.get(), mu_g.get(), std::string{"cf,gc->gfc"})};

    /*
      bisogna capire che fa il k.unsqueeze ?
      B=torch.einsum('gfc,ck->gfk', A , X) * k.unsqueeze(1);
    */
    
    std::unique_ptr<float, CudaDeleter<float>> B {
      (float *)general_einsum(cutensorH,
                              {(int) genes,(int) features, (int)cells}, {(int)cells,(int)features}, A.get(), X.get(), std::string{"gfc,ck->gfk"})}; //l'output ha shape GFF

    
    std::unique_ptr<float, CudaDeleter<float>> Bk {(float *)general_einsum(cutensorH, {(int) genes,(int) features, (int)features}, {(int)genes}, B.get(), k.get(), std::string{"gfc,g->gfc"})}; // ouput ha shape GFF
    B.reset();
    float *vec_inverse;
    CUDA_CHECK(cudaMalloc(&vec_inverse, sizeof(float) * features * features * genes));

    for (int i = 0; i < features * features * genes;
         i = i + features * features) {
      //sto for qui poi lo spignamo bene bene coi strim
      inverseMatrix(cusolverH, Bk.get()+features*features, vec_inverse+features*features, (int) features);
    }
    Bk.reset();
    
    //please check again the shape
    elementWiseSub<<<blocks1D,threads1D>>>(mu_g.get(), genes*cells);
    std::unique_ptr<float, CudaDeleter<float>> C{(float *)general_einsum(
        cutensorH, {(int)cells, (int)features}, {(int)genes, (int)cells},
        X.get(), mu_g.get(), std::string{"cf,gc->gf"})}; // C e' genes*features
                                                         //
    std::unique_ptr<float, CudaDeleter<float>> last{(float *)general_einsum(
        cutensorH, {(int)cells, (int)features}, {(int)genes, (int)cells},
        X.get(), mu_g.get(), std::string{"cf,gc->gf"})}; // C e' genes*features




    // ~1)ora qui faccio elementwise substraction~
    // ~2) poi faccio un esinsum~
    // 3) capire che fa k*C, ma penso si possa fare con un product
    //  in pratica C e' g*f, K invece e' 
    // 4) einsum to calculate delta
    // 5) init_beta
    // 6) norma 2
    // manca moltiplicatione per K da qualche parte
    /*
    //facile, chiamare inversa su tutto B per N volte
      Zigma = torch.inverse(B); // ma e' l'inversa calcolata piu volte, si per
ogni gene !!!
      //transpose e -1 elementwise

      C=torch.einsum("fc,gc->gf", X.t(), (wq_mug - 1));
      //einsum e sgemm
      (genes*1)*(genes*f)
      delta = torch.einsum('gfk,gk->gf', Zigma, k * C)
      
      init_beta += delta //easy
      converged = torch.max(abs(delta)) < eps //usa la norma
     */

  }
}


