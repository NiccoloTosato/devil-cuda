#include <gtest/gtest.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include "inverse.hpp"
#include "cutensor.h"
#include "einsum.hpp"



void invertEye(int cols) {
  float *vec;
  int cols2=cols*cols;
  CUDA_CHECK( cudaMallocManaged(&vec, cols2 *sizeof(float) ) );
  float *vec_inverse;
  CUDA_CHECK( cudaMallocManaged(&vec_inverse, cols2*sizeof(float)) );
  for (int i = 0; i < cols*cols; ++i){
    vec[i] = 0;
    vec_inverse[i]=99;
  }
  for (int i = 0; i < cols; ++i)
    vec[i * cols + i] = 1;
  CUDA_CHECK( cudaDeviceSynchronize());
  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );
  inverseMatrix(cusolverH, vec, vec_inverse, cols);
  CUDA_CHECK( cudaDeviceSynchronize());
  for (int i = 0; i < cols; ++i)
    for(int j=0;j<cols;++j)
      if(i!=j)
        EXPECT_EQ(vec[i*cols+j], 0);
      else
        EXPECT_EQ(vec[i * cols + j], 1);
  CUSOLVER_CHECK( cusolverDnDestroy(cusolverH) );
  CUDA_CHECK( cudaFree(vec) );
  CUDA_CHECK( cudaFree(vec_inverse) );
}

void invertExact(std::vector<float> vec_exact,std::vector<float> vec_inverse_exact,int col) {
  float *vec;
  int col2=col*col;
  CUDA_CHECK(cudaMallocManaged(&vec, col2 * sizeof(float)));
  for (int i = 0; i < col2; ++i)
    vec[i] = vec_exact[i];
  
  float *vec_inverse;
  CUDA_CHECK( cudaMallocManaged(&vec_inverse, col2*sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize());
  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );
  inverseMatrix(cusolverH, vec, vec_inverse, col);
  CUDA_CHECK( cudaDeviceSynchronize());

  for(int i=0;i<col2;++i) 
    EXPECT_NEAR(vec_inverse[i], vec_inverse_exact[i], 1E-6);
  CUSOLVER_CHECK( cusolverDnDestroy(cusolverH) );
  CUDA_CHECK( cudaFree(vec) );
  CUDA_CHECK( cudaFree(vec_inverse) );
  
}

void einSum() {
  cutensorHandle_t handle;
  cutensorCreate(&handle);
  /**********************
   * Setup planCache (optional)
   **********************/
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK( cutensorHandleResizePlanCache(handle, numCachelines) );
  float* output;


    int genes=2;
    int cells=3;
    //init A_device
    float A[]={1,2,3,4,5,6}; // genes x cells
    float* A_device;
    //A=(float *)malloc(genes*cells*sizeof(float));

    CUDA_CHECK(cudaMalloc((void **)&A_device, sizeof(float) * genes * cells));
    cublasSetMatrix(cells, genes, sizeof(float),A, cells, A_device, cells);
    //    printMatrix<<<1,1>>>(A_device,genes,cells);

    //init B_device
    float B[]={1,2,3,4,5,6}; // genes x cells
    float* B_device;
    //B=(float *)malloc(genes*cells*sizeof(float));
    CUDA_CHECK(cudaMalloc((void **)&B_device, sizeof(float) * genes * cells));
    cublasSetMatrix( cells,genes, sizeof(float),B, cells, B_device, cells);

    output= (float*)general_einsum(handle, {2, 3}, {2, 3},A_device,B_device, "fc,gc->gfc"); // contractio>
    //    printMatrix<<<1,1>>>(output,genes*2,cells);

    fflush(stdout);
    cudaDeviceSynchronize();
        // Detach cache and free-up resources
    CUTENSOR_CHECK(cutensorDestroy(handle));
    return 0;
    }
TEST(InvertMatrix, Identity3X3) { invertEye(3); };
TEST(InvertMatrix, Identity100X100) {  invertEye(100); };
TEST(InvertMatrix, Exact2X2) {
  invertExact(std::vector<float>{4.0f, 3.0f, 3.0f, 2.0f},
              std::vector<float>{-2.0f, 3.0f, 3.0f, -4.0f},
              2);
};
TEST(InvertMatrix, Exact3X3) {
  invertExact(
      std::vector<float>{2.0f, 0.0f, -1.0f, 5.0f, 1.0f, 0.0f, 0.0f, 1.0f, 3.0f},
      std::vector<float>{3.0f,  -1.0f, 1.0f,  -15.0f, 6.0f,
                               -5.0f, 5.0f,  -2.0f, 2.0f},
      3);
};




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

