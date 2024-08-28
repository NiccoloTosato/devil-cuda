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
#include <Eigen/Dense>
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
    std::cout << "Loading file " << filename << " Success,elements read: " << data.size() << std::endl;
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

void toGPU(auto vec,float* const vec_gpu) {
  CUDA_CHECK( cudaMemcpy(vec_gpu, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice) );
}

void beta_fit_gpu(Eigen::MatrixXf Y_host, Eigen::MatrixXf X_host, Eigen::MatrixXf mu_beta_host, Eigen::MatrixXf offset_host, Eigen::VectorXf k_host, int max_iter, float eps) {

  /******************************
   * Shape definition 
   ******************************/
  const std::size_t genes{64};
  const std::size_t cells{1024};
  const std::size_t features{2};

  std::size_t genesBatch = 32;
  
  /*******************************
   * Load from disk
   ******************************/
  // const auto X_host = readDatFile("../data/X.dat");
  // const auto Y_host = readDatFile("../data/Y.dat");
  // const auto offset_host = readDatFile("../data/off.dat");
  // const auto mu_beta_host = readDatFile("../data/mu_beta.dat");
  // auto k_host = readDatFile("../data/K.dat");
  
  for (int i=0;i<genes;++i){
    k_host[i] = 1 / k_host[i];
  }
  
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

  std::vector<float*> X(deviceCount);
  std::vector<float*> Y(deviceCount);
  std::vector<float*> offset(deviceCount);
  std::vector<float*> mu_beta(deviceCount);
  std::vector<float *> k(deviceCount);
  std::vector<float *> cg_tmp(deviceCount);
  std::vector<float *> w_q(deviceCount);
  std::vector<float *> mu_g(deviceCount);

  // std::vector<float **> Zigma_pointer(deviceCount);
  // std::vector<float **> Bk_pointer(deviceCount);
  // std::vector<float *> Zigma;
  //this will become again a series of std::vector
  float ***Zigma_pointer = (float***) malloc(sizeof(float **) * deviceCount);
  float ***Bk_pointer = (float***) malloc(sizeof(float **) * deviceCount);
  float** Zigma=(float**)malloc(sizeof(float*) * deviceCount);

  std::vector<float *>    w_qT(deviceCount);
  std::vector<float *>    offsetT(deviceCount);
  std::vector<float *> cg_tmp2(deviceCount);
  std::vector<float *>    A(deviceCount);
  std::vector<float *>    B(deviceCount);
  std::vector<float *>    C(deviceCount);
  std::vector<float *>    Bk(deviceCount);
  std::vector<float *>    delta(deviceCount);
  std::vector<float *>    last(deviceCount);
    
#pragma omp parallel default(shared)//shared(einsum_offsetT,einsum_cg_tmp2,einsum_w_qT,einsum_A,einsum_B,einsum_Bk,einsum_C,einsum_last,einsum_delta,cublasH,cutensorH,Zigma_pointer,Bk_pointer,Zigma,w_qT,offsetT,cg_tmp2,A,B,C,Bk,delta,last,X,Y,offset,k,w_q,mu_g)
  {
    std::size_t BatchCount{genes/genesBatch};

    /****************************
     * Select the device
     ***************************/
    int me{omp_get_thread_num()};
    

    CUDA_CHECK(cudaSetDevice(me));
    /******************************
     * Create handlers and setup
     ******************************/
    CUBLAS_CHECK(cublasCreate(&(cublasH[me])));
    CUTENSOR_CHECK( cutensorCreate( &(cutensorH[me]) ) );
    constexpr int32_t numCachelines = 1024;
    CUTENSOR_CHECK( cutensorHandleResizePlanCache(cutensorH[me], numCachelines) );
    /********************************
     * Allocate and copy X on each device, since it is const do it now
     *******************************/
    CUDA_CHECK( cudaMalloc((void**)&X[me], features*cells*sizeof(float)) );
    toGPU(X_host, X[me]);
    /*********************************
     * Allocate Y,offset,K,mu_beta, but use genesBatch as size, not genes
     ********************************/
    CUDA_CHECK( cudaMalloc((void**)&Y[me], cells*genesBatch*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&offset[me], genesBatch*cells*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&mu_beta[me], genesBatch*features*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&k[me], genesBatch*sizeof(float)) );
    //CUDA_CHECK( cudaMalloc((void**)&cg_tmp[me], genesBatch*cells*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&w_q[me], genesBatch*cells*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&mu_g[me], genesBatch*cells*sizeof(float)) );

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
				    {(int)genesBatch});
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
    w_qT[me]=einsum_w_qT[me].allocate();
    offsetT[me] = einsum_offsetT[me].allocate();
    cg_tmp2[me] = einsum_cg_tmp2[me].allocate();
    A[me] = einsum_A[me].allocate();
    B[me] = einsum_B[me].allocate();
    C[me] = einsum_C[me].allocate();
    Bk[me] = einsum_Bk[me].allocate();
    delta[me] = einsum_delta[me].allocate();
    last[me] = einsum_last[me].allocate();
    //free offset memory after transposition
    //offset.reset();
    /******************************
     * Allocate Zigma, The array of pointer to Zigma and Bk
     ******************************/

    // Use Managed memory to simply set the addresses
    CUDA_CHECK(cudaMallocManaged((void **) &(Zigma_pointer[me]), genesBatch * sizeof(float*)) );
    CUDA_CHECK(cudaMallocManaged((void **) &(Bk_pointer[me]), genesBatch * sizeof(float*)) );
    CUDA_CHECK(cudaMalloc((void **) &(Zigma[me]), sizeof(float) * features * features * genesBatch));

    for (int i = 0; i < genesBatch; ++i) {
      Zigma_pointer[me][i] = Zigma[me] + features * features * i;
      Bk_pointer[me][i] = Bk[me] + features * features * i;
    }

    cudaDeviceSynchronize();
    
#pragma omp single
    { //here we will generate the work ! 
      for (int i = 0; i < BatchCount; ++i) {
#pragma omp task default(shared)//shared(X,Y,offset,mu_beta,k,cg_tmp,mu_g,w_q,offset_host)
        {
          std::cout << "Batch " << i << " computed by " << omp_get_thread_num()
                    << std::endl;
	  int me=omp_get_thread_num();
          // copy the necessary data!
          CUDA_CHECK(cudaMemcpy(
              offset[me],
              offset_host.data() + i  * genesBatch * cells,
              genesBatch * cells * sizeof(float), cudaMemcpyHostToDevice));
          einsum_offsetT[me].execute(cutensorH[me], offset[me], nullptr);
          CUDA_CHECK(cudaMemcpy(
			     mu_beta[me], mu_beta_host.data() +  i *genesBatch * features,
			     genesBatch*features* sizeof(float),
			     cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(
				k[me],  k_host.data() +  i *genesBatch*1,
				genesBatch*1* sizeof(float),
                                cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(
              Y[me], Y_host.data() +  i * genesBatch * cells ,
              genesBatch * cells * sizeof(float), cudaMemcpyHostToDevice));
	  //set something to zero, required ? BOH,sicuro non falliremo per sta cosa qui
	  CUDA_CHECK( cudaMemset(w_q[me], 0, genesBatch * cells * sizeof(float)));
	  CUDA_CHECK( cudaMemset(mu_g[me], 0, genesBatch*cells*sizeof(float)));
          // execute the computation
	  /******************************
	   * Initialize norm s.t. the initial check is always True , set iter to 0,
	   * measure start time.
	   ******************************/
	  //I know, there is a narrow conversion here.
          float norm(std::sqrt(genesBatch * features));
          std::size_t iter{0};
	  auto t1 = std::chrono::high_resolution_clock::now();
	  while (iter < 100 && (norm/std::sqrt(genesBatch*features)) > 1E-4) {
	    ++iter;
            einsum_cg_tmp2[me].execute(cutensorH[me], X[me], mu_beta[me]);
	    dim3 threads1D(256);
	    dim3 blocks1D((genesBatch * cells + threads1D.x - 1) / threads1D.x);
	    expGPU<<<blocks1D, threads1D>>>(cg_tmp2[me], offsetT[me], w_q[me],
					    genesBatch * cells);
            einsum_w_qT[me].execute(cutensorH[me], w_q[me], nullptr);

	    dim3 threads2D(16,16);
	    dim3 blocks2D((cells + threads2D.x - 1) / threads2D.x,
			  (genesBatch + threads2D.y - 1) / threads2D.y);
	    process2D<<<blocks2D, threads2D>>>(k[me], Y[me], w_qT[me], mu_g[me],
					       genesBatch, cells);
            elementWise<<<blocks1D, threads1D>>>(mu_g[me], w_qT[me],
                                                 genesBatch * cells);
            einsum_A[me].execute(cutensorH[me], X[me], mu_g[me]);
            einsum_B[me].execute(cutensorH[me], A[me], X[me]);
            einsum_Bk[me].execute(cutensorH[me], B[me], k[me]);
            inverseMatrix2(cublasH[me], Bk_pointer[me], Zigma_pointer[me],
                           features, genesBatch);

	    elementWiseSub<<<blocks1D,threads1D>>>(mu_g[me], genesBatch*cells);
	    einsum_C[me].execute(cutensorH[me], X[me], mu_g[me]);
            einsum_last[me].execute(cutensorH[me], k[me], C[me]);
            einsum_delta[me].execute(cutensorH[me], Zigma[me], last[me]);
	    final1D<<<blocks1D,threads1D>>>(mu_beta[me],delta[me],genesBatch*features);
            cublasSnrm2(cublasH[me], genesBatch * features, delta[me], 1,
                        &norm);
	  }
	  auto t2 = std::chrono::high_resolution_clock::now();
	  auto elapsed{t2-t1};
          std::cout
              << std::chrono::duration<double, std::milli>(elapsed).count() /
                     iter
              << " ms [avg iter time]" << std::endl;
	  std::cout << "mu_beta {"<<genesBatch<<","<<features <<"}\n";
	  printMatrix<<<1, 1>>>( genesBatch,features, mu_beta[me]);
	  cudaDeviceSynchronize();
	  std::cout << std::flush;
            // copy back the data, this assume that I prepared something!
        }
      }
    }
    // free the memory
    CUDA_CHECK(cudaFree(Zigma[me]));
    CUDA_CHECK(cudaFree(Bk_pointer[me]));
    CUDA_CHECK(cudaFree(Zigma_pointer[me]));
    CUDA_CHECK(cudaFree(w_qT[me]));
    CUDA_CHECK(cudaFree(offsetT[me]));
    CUDA_CHECK(cudaFree(cg_tmp2[me]));
    CUDA_CHECK(cudaFree(A[me]));
    CUDA_CHECK(cudaFree(B[me]));
    CUDA_CHECK(cudaFree(C[me]));
    CUDA_CHECK(cudaFree(Bk[me]));
    CUDA_CHECK(cudaFree(delta[me]));
    CUDA_CHECK(cudaFree(last[me]));

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


