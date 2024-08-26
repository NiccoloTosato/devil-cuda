#include <cmath>
#include <cstddef>
#include <cstdio>
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
#include <thread>
#include "einsum.hpp"
#include "kernel.h"
#include "inverse.hpp"
#include "utils.hpp"
#include "cutensor.h"
#include <omp.h>

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
  /* small debug files in ../data-debug/
  std::size_t genes{3};
  std::size_t cells{4};
  std::size_t features{2};
  */
  /******************************
   * Shape definition 
   ******************************/
  const std::size_t genes{64};
  const std::size_t cells{1024};
  const std::size_t features{2};
  std::size_t batchSize = 8;
  /*******************************
   * Load from disk
   ******************************/
  const auto X_host = readDatFile("../data/X.dat");
  const auto Y_host = readDatFile("../data/Y.dat");
  const auto offset_host = readDatFile("../data/off.dat");
  const auto mu_beta_host = readDatFile("../data/mu_beta.dat");
  auto k_host = readDatFile("../data/K.dat");
  for (auto &x : k_host)
    x=1/x;
  //  const auto mu_beta_host = readDatFile("../data/mu_beta.dat");

  std::vector<float> mu_beta_final(genes*features, 0.0);
	    
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  
  std::cout << "Device " << 0 << ": " << deviceProp.name << std::endl;
  std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
  std::cout << "X {"<<cells<<","<<features <<"}\n";
  std::cout << "Y {" << genes << "," << cells << "}\n";
  std::cout << "offset {"<<genes<<","<<cells <<"}\n";
  std::cout << "mu_beta {" << genes << "," << features << "}\n";
  std::cout << "K {" << genes << "," << 1 << "}\n";

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "Detected " << deviceCount << " GPU(s)" << std::endl;
  omp_set_num_threads(deviceCount);


  /************************
   * Create array of handles, variables, pointer, object and so on. 
   ***********************/
  std::vector<cublasHandle_t> cublasH(deviceCount);
  std::vector<cutensorHandle_t> cutensorH(deviceCount);

  //this will call a dummy constructor, don't worry about initialization!
  std::vector<EinsumWrapper> einsum_offsetT(deviceCount);
  std::vector<EinsumWrapper> einsum_cg_tmp2(deviceCount);
  std::vector<EinsumWrapper> einsum_w_qT(deviceCount);
  std::vector<EinsumWrapper> einsum_A(deviceCount);
  std::vector<EinsumWrapper> einsum_B(deviceCount);
  std::vector<EinsumWrapper> einsum_Bk(deviceCount);
  std::vector<EinsumWrapper> einsum_C(deviceCount);
  std::vector<EinsumWrapper> einsum_last(deviceCount);
  std::vector<EinsumWrapper> einsum_delta(deviceCount);

  std::vector<float *> Y(deviceCount);
  std::vector<float*> offset(deviceCount);
  std::vector<float*> Y(deviceCount);
#pragma omp parallel shared(einsum_offsetT,einsum_cg_tmp2,einsum_w_qT,einsum_A,einsum_B,einsum_Bk,einsum_C,einsum_last,einsum_delta,cublasH,cutensorH)
  {
    std::size_t genesBatch{genes / batchSize};
    /****************************
     * Select the device
     ***************************/
    int me{omp_get_thread_num()};
    CUDA_CHECK(cudaSetDevice(me));
    /******************************
     * Create handlers and setup
     ******************************/
    CUBLAS_CHECK(cublasCreate(&(cublasH[me])));
    {
      CUTENSOR_CHECK( cutensorCreate( &(cutensorH[me]) ) );
    }
    constexpr int32_t numCachelines = 1024;
    CUTENSOR_CHECK( cutensorHandleResizePlanCache(cutensorH[me], numCachelines) );
    /********************************
     * Allocate X on each device, since it is const do it only once
     *******************************/
    float* tmp=nullptr;
    CUDA_CHECK( cudaMalloc((void**)&tmp, features*cells*sizeof(float)) );
    std::unique_ptr<float,CudaDeleter<float>> X{tmp};
    toGPU(X_host, X.get());
    /*********************************
     * Allocate Y,offset,K,mu_beta, but use genesBatch as size, not genes
     ********************************/
    CUDA_CHECK( cudaMalloc((void**)&tmp, cells*genesBatch*sizeof(float)) );
    std::unique_ptr<float,CudaDeleter<float>> Y{tmp};
    CUDA_CHECK( cudaMalloc((void**)&tmp, genesBatch*cells*sizeof(float)) );
    std::unique_ptr<float,CudaDeleter<float>> offset{tmp};
    CUDA_CHECK( cudaMalloc((void**)&tmp, genesBatch*features*sizeof(float)) );
    std::unique_ptr<float, CudaDeleter<float>> mu_beta{tmp};
    CUDA_CHECK( cudaMalloc((void**)&tmp, genesBatch*sizeof(float)) );
    std::unique_ptr<float,CudaDeleter<float>> k{tmp};
    CUDA_CHECK( cudaMalloc((void**)&tmp, genesBatch*cells*sizeof(float)) );
    std::unique_ptr<float, CudaDeleter<float>> cg_tmp{tmp};
    CUDA_CHECK( cudaMalloc((void**)&tmp, genesBatch*cells*sizeof(float)) );
    std::unique_ptr<float,CudaDeleter<float>> w_q{tmp};
    CUDA_CHECK( cudaMalloc((void**)&tmp, genesBatch*cells*sizeof(float)) );
    std::unique_ptr<float,CudaDeleter<float>> mu_g{tmp};
    /*********************************
     * Initialize the Tensor object, this doesn't allocate nothing ! 
     ********************************/
    einsum_offsetT[me] = EinsumWrapper(std::string{"ij->ji"},
                                       {(int)genesBatch, (int)cells},
				       {});

    einsum_cg_tmp2[me] = EinsumWrapper(std::string{"ik,jk->ij"},
                                       {(int)cells, (int)features},
				       {(int)genesBatch, (int)features});
    einsum_w_qT[me] = EinsumWrapper( std::string{"ij->ji"},
				     {(int)cells, (int)genesBatch},
			      {});
    einsum_A[me] = EinsumWrapper(std::string{"cf,gc->gfc"},
                                 {(int)cells, (int)features},
				 {(int)genesBatch, (int)cells});
    einsum_B[me] = EinsumWrapper( std::string{"gfc,ck->gfk"},
			  {(int)genesBatch, (int)features, (int)cells},
				  {(int)cells, (int)features});
    einsum_Bk[me] = EinsumWrapper ( std::string{"gfc,g->gfc"},
                          {(int)genesBatch, (int)features, (int)features},
				    {(int)genes});
    einsum_C[me] = EinsumWrapper ( std::string{"cf,gc->gf"},
			  {(int)cells, (int)features},
				   {(int)genesBatch, (int)cells});
    einsum_last[me] = EinsumWrapper ( std::string{"gk,gf->gf"},
			      {(int)genesBatch, 1},
				      {(int)genesBatch, (int)features});
    einsum_delta[me] =
        EinsumWrapper(std::string{"gfk,gk->gf"},
                      {(int)genesBatch, (int)features, (int)features},
                      {(int)genesBatch, (int)features});

    /******************************
     * This allocate the workspace and the output tensor
     ******************************/
    float *w_qT=einsum_w_qT[me].allocate();
    float *offsetT = einsum_offsetT[me].allocate();
    float *cg_tmp2 = einsum_cg_tmp2[me].allocate();
    float *A = einsum_A[me].allocate();
    float *B = einsum_B[me].allocate();
    float *C = einsum_C[me].allocate();
    float* Bk = einsum_Bk[me].allocate();
    float *delta = einsum_delta[me].allocate();
    float *last = einsum_last[me].allocate();
    
    //free offset memory after transposition
    //offset.reset();
    /******************************
     * Allocate Zigma, The array of pointer to Zigma and Bk
     ******************************/
    float **Zigma_pointer;
    float **Bk_pointer; 
    float *Zigma;
    //Use Managed memory to simply set the addresses
    CUDA_CHECK(cudaMallocManaged(&Zigma_pointer, genesBatch * sizeof(float*)) );
    CUDA_CHECK(cudaMallocManaged(&Bk_pointer, genesBatch * sizeof(float*)) );
    CUDA_CHECK(cudaMalloc(&Zigma, sizeof(float) * features * features * genesBatch));
    for (int i = 0; i < genesBatch; ++i) {
      Zigma_pointer[i] = Zigma + features * features * i;
      Bk_pointer[i] = Bk + features * features * i;
    }
    cudaDeviceSynchronize();
    
#pragma omp single nowait
    { //here we will generate the work ! 
      for (int i = 0; i < genes / batchSize; ++i) {
#pragma omp task shared(X,Y,offset,mu_beta,k,cg_tmp,mu_g,w_q,offset_host)
        {
          std::cout << "Batch " << i << " computed by " << omp_get_thread_num()
                    << std::endl;
	  int me=omp_get_thread_num();
          std::cout << "Offset address " << offset.get() << std::endl;
	  std::this_thread::sleep_for(std::chrono::seconds(8));
          // copy the necessary data!

          CUDA_CHECK(cudaMemcpy(
				offset.get(),
				offset_host.data() + i * genesBatch * cells * sizeof(float),
				genesBatch * cells * sizeof(float), cudaMemcpyHostToDevice));
          einsum_offsetT[me].execute(cutensorH[me], offset.get(), nullptr);
	  
          CUDA_CHECK(cudaMemcpy(
				mu_beta.get(), mu_beta_host.data() + i*genesBatch * features * sizeof(float),
				genesBatch*features* sizeof(float),
                                cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(
				k.get(), k_host.data() + i*genesBatch * 1 * sizeof(float),
				genesBatch*1* sizeof(float),
                                cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(
				Y.get(), Y_host.data() + i*genesBatch * cells * sizeof(float),
				genesBatch*cells* sizeof(float),
                                cudaMemcpyHostToDevice));

	  //set something to zero, required ? BOH
          CUDA_CHECK( cudaMemset(cg_tmp.get(), 0, genesBatch*cells*sizeof(float)));
	  CUDA_CHECK( cudaMemset(w_q.get(), 0, genesBatch * cells * sizeof(float)));
	  CUDA_CHECK( cudaMemset(mu_g.get(), 0, genesBatch*cells*sizeof(float)));
          // execute the computation
	  /******************************
	   * Initialize norm s.t. the initial check is always True , set iter to 0,
	   * measure start time.
	   ******************************/
	  //I know, there is a narrow conversion here.
	  float norm(std::sqrt(genesBatch*features));
	  std::size_t iter{0};
	  auto t1 = std::chrono::high_resolution_clock::now();
	  while (iter < 8000 && (norm/std::sqrt(genesBatch*features)) > 1E-12) {
	    ++iter;
	    einsum_cg_tmp2[me].execute(cutensorH[me], X.get(), mu_beta.get());
	    dim3 threads1D(256);
	    dim3 blocks1D((genesBatch * cells + threads1D.x - 1) / threads1D.x);
	    expGPU<<<blocks1D, threads1D>>>(cg_tmp2, offsetT, w_q.get(),
					    genesBatch * cells);
	    einsum_w_qT[me].execute(cutensorH[me],w_q.get(),nullptr);
	    dim3 threads2D(16,16);
	    dim3 blocks2D((cells + threads2D.x - 1) / threads2D.x,
			  (genesBatch + threads2D.y - 1) / threads2D.y);
	    process2D<<<blocks2D, threads2D>>>(k.get(), Y.get(), w_qT, mu_g.get(),
					       genesBatch, cells);
	    elementWise<<<blocks1D, threads1D>>>(mu_g.get(), w_qT, genesBatch * cells);
	    einsum_A[me].execute(cutensorH[me], X.get(), mu_g.get());
	    einsum_B[me].execute(cutensorH[me], A, X.get());
	    einsum_Bk[me].execute(cutensorH[me],B,k.get());
	    inverseMatrix2(cublasH[me], Bk_pointer, Zigma_pointer, features ,genesBatch);
	    elementWiseSub<<<blocks1D,threads1D>>>(mu_g.get(), genesBatch*cells);
	    einsum_C[me].execute(cutensorH[me],X.get(),mu_g.get()); 
	    einsum_last[me].execute(cutensorH[me],k.get(),C);
	    einsum_delta[me].execute(cutensorH[me],Zigma,last);
	    final1D<<<blocks1D,threads1D>>>(mu_beta.get(),delta,genesBatch*features);
	    cublasSnrm2(cublasH[me], genes * features, delta, 1, &norm);
	  }
	  auto t2 = std::chrono::high_resolution_clock::now();
	  auto elapsed{t2-t1};
	  std::cout<<std::chrono::duration<double,std::milli>(elapsed).count() / iter << " ms [avg iter time]"<<std::endl;
            // copy back the data, this assume that I prepared something!
        }
      }
    }
    // free the memory
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
    //CUBLAS_CHECK( cublasDestroy(cublasH) );
    //CUTENSOR_CHECK( cutensorDestroy(cutensorH) );
  }
  
  //  std::cout << "mu_beta {"<<genes<<","<<features <<"}\n";
  //  printMatrix<<<1, 1>>>( genes,features, mu_beta.get());
  //  cudaDeviceSynchronize();

  //std::cout << "Norm " << norm / std::sqrt(genes * features) << std::endl;


}


