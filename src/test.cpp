#include <gtest/gtest.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
#include <vector>
#include "utils.hpp"
#include "inverse.hpp"
#include "cutensor.h"
#include "einsum.hpp"
#include  <string>

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

float* einSum(std::vector<float> a, std::vector<int> a_shape,
            std::vector<float> b, std::vector<int> b_shape,
            std::string s) {
  
  cutensorHandle_t handle;
  cutensorCreate(&handle);
  /**********************
   * Setup planCache (optional)
   **********************/
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK(cutensorHandleResizePlanCache(handle, numCachelines));
  //the output allocation is managed by einsum!
  float* output;

  float *a_device;
  int size_a = 1;
  for (auto dim : a_shape) {
    size_a=size_a*dim;
  }
  int size_b = 1;
  for (auto dim : b_shape) {
    size_b=size_b*dim;
  }
  CUDA_CHECK( cudaMallocManaged(&a_device, size_a * sizeof(float)));
  for (int i = 0; i < size_a; ++i)
    a_device[i] = a[i];
  
  float *b_device;
  CUDA_CHECK( cudaMallocManaged(&b_device, size_b*sizeof(float)) );
  for (int i = 0; i < size_b; ++i)
    b_device[i]=b[i];
  output =(float*) general_einsum(handle, a_shape, b_shape, a_device,b_device, s.c_str()); 
  cudaDeviceSynchronize();
  CUTENSOR_CHECK(cutensorDestroy(handle));
  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  return output;
}


TEST(EinSum, Test1) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {120.0f, 165.0f, 168.0f,
                                     231.0f, 216.0f, 297.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ei,jk->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Test2) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {7.0f, 16.0f,27.0f,40.0f,55.0f,72.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ij,ij->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
  CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSum, Test3) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {144.0f, 495.0f};
  
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ij,ik->i"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Test4) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {50.0f, 122.0f,
                                     68.0f, 167.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"fc,gc->gf"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Test5) {
    std::vector<float> a = {
      1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,
      1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    std::vector<int> a_shape = {2,2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {50.0f, 122.0f,
                                     68.0f, 167.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ijk,ik->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};


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

