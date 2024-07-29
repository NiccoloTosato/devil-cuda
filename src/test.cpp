#include <gtest/gtest.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.hpp"
#include "inverse.hpp"

void invert(int cols) {
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

TEST(InvertMatrix, Identity3X3) { invertEye(3); };
TEST(InvertMatrix, Identity100X100) {  invertEye(100); };

TEST(InvertMatrix, Exact2X2) {
  float *vec;
  float vec_exact[] = {3.0f, 4.0f, 1.0f, 2.0f};
  CUDA_CHECK(cudaMallocManaged(&vec, 4 * sizeof(float)));
  for (int i = 0; i < 4; ++i)
    vec[i] = vec_exact[i];
  
  float *vec_inverse;
  float vec_inverse_exact[]={1.0f,-2.0f,-0.5f,1.5f};
  CUDA_CHECK( cudaMallocManaged(&vec_inverse, 4*sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize());
  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );
  inverseMatrix(cusolverH, vec, vec_inverse, 2);
  CUDA_CHECK( cudaDeviceSynchronize());

  for(int i=0;i<4;++i) 
    EXPECT_NEAR(vec_inverse[i], vec_inverse_exact[i], 1E-6);

  CUSOLVER_CHECK( cusolverDnDestroy(cusolverH) );
  CUDA_CHECK( cudaFree(vec) );
  CUDA_CHECK( cudaFree(vec_inverse) );
};

TEST(InvertMatrix, Exact3X3) {
  float *vec;
  float vec_exact[]={2.0f,0.0f,-1.0f,5.0f,1.0f,0.0f,0.0f,1.0f,3.0f};
  CUDA_CHECK( cudaMallocManaged(&vec, 9 *sizeof(float) ) );
  for (int i = 0; i < 9; ++i)
    vec[i] = vec_exact[i];

  float *vec_inverse;
  float vec_inverse_exact[] = {3.0f,  -1.0f, 1.0f,  -15.0f, 6.0f,
                               -5.0f, 5.0f,  -2.0f, 2.0f};
  
  CUDA_CHECK( cudaMallocManaged(&vec_inverse, 9*sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize());
  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK( cusolverDnCreate(&cusolverH) );
  inverseMatrix(cusolverH, vec, vec_inverse, 3);
  CUDA_CHECK( cudaDeviceSynchronize());
  for (int i = 0; i < 9; ++i) 
    EXPECT_NEAR(vec_inverse[i], vec_inverse_exact[i], 1E-6);
  CUSOLVER_CHECK( cusolverDnDestroy(cusolverH) );
  CUDA_CHECK( cudaFree(vec) );
  CUDA_CHECK( cudaFree(vec_inverse) );
};


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

