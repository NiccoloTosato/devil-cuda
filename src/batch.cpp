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

__global__ void printMatrix(const int rows,const int cols, float* const matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
	  printf("%2.4f ", matrix[j*rows+i]);
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

//Eigen::MatrixXf beta_fit_gpu_external(Eigen::MatrixXf Y_host, Eigen::MatrixXf X_host, Eigen::MatrixXf mu_beta_host, Eigen::MatrixXf offset_host, Eigen::VectorXf k_host, int max_iter, float eps) {
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
beta_fit_gpu_external(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const &
        Y_host,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const &
        X_host,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const &
        mu_beta_host,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const &
        offset_host,
    Eigen::VectorXf const & kk_host, int max_iter, float eps,int batch_size) {
  
  /******************************
   * Shape definition 
   ******************************/
  const std::size_t genes(Y_host.cols());
  const std::size_t cells(X_host.cols());
  const std::size_t features(X_host.rows());

  std::cout << "X {"<<X_host.rows()<<","<<X_host.cols() <<"}\n";
  std::cout << "Y {" << Y_host.rows() << "," << Y_host.cols() << "}\n";
  std::cout << "offset {"<<offset_host.rows()<<","<<offset_host.cols() <<"}\n";
  std::cout << "mu_beta {" << mu_beta_host.rows()  << "," << mu_beta_host.cols() << "}\n";
  std::cout << "K {" << kk_host.size() << "," << 1 << "}\n";
  std::cout << "Genes" << genes <<std::endl;
  std::cout << "Cells" << cells <<std::endl;
  std::cout << "Features" << features <<std::endl;
  std::size_t genesBatch = batch_size;
  // std::cout << offset_host << std::endl;
    /*******************************
   * Load from disk
   ******************************/
  /*
  std::cout << "OUTPUT OFFSET " << std::endl;
  std::cout << "OUTPUT {"<<offset_host.rows()<<","<<offset_host.cols()<<"}" <<
std::endl; std::cout << "OUTPUT OFFSET " << std::endl; for (int i = 0; i < cells
; ++i) { for (int j = 0; j < genesBatch;++j) { std::cout <<
offset_host.data()+j*cells+i << " " ;
    }
    std::cout << std::endl;
  }
  */
  Eigen::VectorXf k_host(kk_host.size());
  for (int i=0;i<genes;++i){

    k_host[i] = 1 / kk_host[i];



  }
  
  //  const auto mu_beta_host = readDatFile("../data/mu_beta.dat");

  std::vector<float> mu_beta_final(genes*features, 0.0);
	    
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "Device " << 0 << ": " << deviceProp.name << std::endl;
  std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

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
  //std::vector<EinsumWrapper> einsum_offsetT(deviceCount);
  std::vector<EinsumWrapper> einsum_cg_tmp2(deviceCount);
  //std::vector<EinsumWrapper> einsum_w_qT(deviceCount);
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

  //std::vector<float *>    w_qT(deviceCount);
  //std::vector<float *>    offsetT(deviceCount);
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
    //einsum_offsetT[me] =
    //    EinsumWrapper(std::string{"ij->ji"}, {(int)cells, (int)genesBatch},
    //                  {}); // NON SERVE PIU
    einsum_cg_tmp2[me] = EinsumWrapper(std::string{"ik,jk->ji"},
                                       {(int)cells, (int)features},
				       {(int)genesBatch, (int)features}); //E" CORRETTO
    //einsum_w_qT[me] = EinsumWrapper( std::string{"ij->ji"},
    //				     {(int)genesBatch, (int)cells},
    //				     {}); // NON SERVE PIU!!!!!!!!!!!!!!
    einsum_A[me] = EinsumWrapper(std::string{"cf,gc->cfg"},
                                 {(int)cells, (int)features},
				 {(int)genesBatch, (int)cells}); // ASSUMIAMO CHE SIA CORRETTO COSI, HYP
    einsum_B[me] = EinsumWrapper( std::string{"cfg,ck->gkf"},
			  {(int)cells, (int)features, (int)genesBatch},
				  {(int)cells, (int)features});  // ASSUMIAMO CHE SIA CORRETTO COSI, HYP
    einsum_Bk[me] = EinsumWrapper ( std::string{"gfc,g->gfc"},
                          {(int)genesBatch, (int)features, (int)features},
				    {(int)genesBatch}); // ASSUMIAMO CORRETTO
    einsum_C[me] = EinsumWrapper ( std::string{"cf,gc->gf"},
			  {(int)cells, (int)features}, 
				   {(int)genesBatch, (int)cells}); // ASSUMIAMO CORRETTO
    einsum_last[me] = EinsumWrapper ( std::string{"g,gf->gf"},
			      {(int)genesBatch},
				      {(int)genesBatch, (int)features}); //ok
    einsum_delta[me] =
      EinsumWrapper(std::string{"gfk,gk->gf"},
                      {(int)genesBatch, (int)features, (int)features},
                      {(int)genesBatch, (int)features});

    /******************************
     * This allocate the workspace and the output tensor
     ******************************/
    //w_qT[me]=einsum_w_qT[me].allocate();
    //offsetT[me] = einsum_offsetT[me].allocate();
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
    {
      std::size_t free_mem, total_mem;

    // Get the amount of free and total memory
      CUDA_CHECK( cudaMemGetInfo(&free_mem, &total_mem) );
      std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB"
                << std::endl;
      std::cout << "Used memory: " << (total_mem-free_mem) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

    }
#pragma omp single
    { //here we will generate the work ! 
      for (int i = 0; i < BatchCount; ++i) {
#pragma omp task default(shared)//shared(X,Y,offset,mu_beta,k,cg_tmp,mu_g,w_q,offset_host)
        {
	  //          std::cout << "Batch " << i << " computed by " << omp_get_thread_num()
          //          << std::endl;
	  int me=omp_get_thread_num();
          // copy the necessary data!
          CUDA_CHECK(cudaMemcpy(
              offset[me],
              offset_host.data() + i  * genesBatch * cells,
              genesBatch * cells * sizeof(float), cudaMemcpyHostToDevice)); //CORRETTO
	  //einsum_offsetT[me].execute(cutensorH[me], offset[me], nullptr);

          CUDA_CHECK(cudaMemcpy(
              mu_beta[me], mu_beta_host.data() + i * genesBatch * features,
              genesBatch * features * sizeof(float), cudaMemcpyHostToDevice)); //CORRETTO
	  /*
            if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "mu_beta {"<<features<<","<< genesBatch <<"}\n";
              printMatrix<<<1, 1>>>(features, genesBatch, mu_beta[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } 
	  */


          CUDA_CHECK(cudaMemcpy(k[me], k_host.data() + i * genesBatch * 1,
                                genesBatch * 1 * sizeof(float),
                                cudaMemcpyHostToDevice)); // CORRETTO

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
	  while (iter < max_iter && (norm/std::sqrt(genesBatch*features)) > eps) {
            ++iter;
            einsum_cg_tmp2[me].execute(cutensorH[me], X[me], mu_beta[me]);
	    dim3 threads1D(256);
	    dim3 blocks1D((genesBatch * cells + threads1D.x - 1) / threads1D.x);
            expGPU<<<blocks1D, threads1D>>>(cg_tmp2[me], offset[me], w_q[me],
                                            genesBatch * cells);
            /*
            if(me==0) {
          cudaDeviceSynchronize();
          std::cout << ",cg_tmp2 {"<<cells<<","<< genesBatch <<"}\n";
          printMatrix<<<1, 1>>>(cells, genesBatch, cg_tmp2[me]);
          cudaDeviceSynchronize();
          std::cout << std::fflush;

            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
            }

            if(me==0) {
              cudaDeviceSynchronize();
              std::cout << ",w_q {"<<cells<<","<< genesBatch <<"}\n";
              printMatrix<<<1, 1>>>(cells, genesBatch, w_q[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
            }
            */
	    dim3 threads2D(16,16);
	    dim3 blocks2D((cells + threads2D.x - 1) / threads2D.x,
			  (genesBatch + threads2D.y - 1) / threads2D.y); // STILL CONTINUA A FUNZIONARE
            process2D<<<blocks2D, threads2D>>>(k[me], Y[me], w_q[me],
                                               mu_g[me], // NON VA CAMBIATO NADA FUNZIONA BENISSIMO
                                               genesBatch, cells);


            elementWise<<<blocks1D, threads1D>>>(mu_g[me], w_q[me],
                                                 genesBatch * cells); //FUNZIONA ANCHE IN Forder
	    /*
            if(me==0) {
              cudaDeviceSynchronize();
              std::cout << ",mu_g {"<<cells<<","<< genesBatch <<"}\n";
              printMatrix<<<1, 1>>>(cells, genesBatch, mu_g[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } */

            einsum_A[me].execute(cutensorH[me], X[me], mu_g[me]);
	    /*
            if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "A[0] {"<<genesBatch<<","<< features <<"}\n";
              printMatrix<<<1, 1>>>(genesBatch,features, A[me]+genesBatch*features);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } 

	    */
            einsum_B[me].execute(cutensorH[me], A[me], X[me]);
	    /*
            if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "B[0] {"<<features<<","<< features <<"}\n";
              printMatrix<<<1, 1>>>(features,features, B[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } 
	    */
            einsum_Bk[me].execute(cutensorH[me], B[me], k[me]);
	    /*
            if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "Bk[0] {"<<features<<","<< features <<"}\n";
              printMatrix<<<1, 1>>>(features,features, Bk[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } 
	    */
            inverseMatrix2(cublasH[me], Bk_pointer[me], Zigma_pointer[me],
                           features, genesBatch);
            /*
            if (me == 0) {
              //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "Zigma[0] {"<<features<<","<< features <<"}\n";
              printMatrix<<<1, 1>>>(features,features, Zigma_pointer[me][1]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
                  }
	    */
	    //FINO QUI FUNZIONA, 1/9/2024
	    elementWiseSub<<<blocks1D,threads1D>>>(mu_g[me], genesBatch*cells);
            einsum_C[me].execute(cutensorH[me], X[me], mu_g[me]);
	    /*
            if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "C {"<<features<<","<< genesBatch <<"}\n";
              printMatrix<<<1, 1>>>(features,genesBatch, C[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } 
	    */
            einsum_last[me].execute(cutensorH[me], k[me], C[me]);
	    /*
            if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "last {"<<features<<","<< genesBatch <<"}\n";
              printMatrix<<<1, 1>>>(features,genesBatch, last[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  }
	    */
            einsum_delta[me].execute(cutensorH[me], Zigma[me], last[me]);
	    /*
	    if (me == 0) {
	      //fare funzione per printare direttamente A
              cudaDeviceSynchronize();
              std::cout << "delta {"<<features<<","<< genesBatch <<"}\n";
              printMatrix<<<1, 1>>>(features,genesBatch, delta[me]);
              cudaDeviceSynchronize();
              std::cout << std::fflush;
            } else {
                  std::this_thread::sleep_for(std::chrono::seconds(15));
		  } 
	    */

            final1D<<<blocks1D, threads1D>>>(mu_beta[me], delta[me],
                                             genesBatch * features);
            cublasSnrm2(cublasH[me], genesBatch * features, delta[me], 1,
                        &norm);
	  }
	  auto t2 = std::chrono::high_resolution_clock::now();
	  auto elapsed{t2-t1};
/*          std::cout
              << std::chrono::duration<double, std::milli>(elapsed).count() /
                     iter
              << " ms [avg iter time]" << std::endl;
	  //	  std::cout << " iter " << iter << "\nmu_beta {"<<genesBatch<<","<<features <<"}\n";
	  //   printMatrix<<<1, 1>>>(features, genesBatch, mu_beta[me]);
	  //   std::cout << std::flush;
*/
          cudaDeviceSynchronize();
	  //	  std::this_thread::sleep_for(std::chrono::seconds(150));
          CUDA_CHECK(cudaMemcpy(mu_beta_final.data() +  i *genesBatch * features,
			     mu_beta[me], 
			     genesBatch*features* sizeof(float),
			     cudaMemcpyDeviceToHost));
          cudaDeviceSynchronize();
            // copy back the data, this assume that I prepared something!
        }
      }
    }
    // free the memory
    CUDA_CHECK(cudaFree(Zigma[me]));
    CUDA_CHECK(cudaFree(Bk_pointer[me]));
    CUDA_CHECK(cudaFree(Zigma_pointer[me]));
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
  


  //std::cout << "Norm " << norm / std::sqrt(genes * features) << std::endl;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> result(mu_beta_final.data(), features,genes);
  return result;
}

void hello() {
  std::cout<<"Linking okay"<<std::endl;
}
