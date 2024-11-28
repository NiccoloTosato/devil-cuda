#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>
#include <math.h>
#include "utils.hpp"

#include <Eigen/Dense>
template<typename T>
void toGPU(T vec,float* const vec_gpu) {
  CUDA_CHECK( cudaMemcpy(vec_gpu, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice) );
}

// Kernel for element-wise operation log1p(y / exp(offset_matrix))
__global__ void compute_log1p(const float* y, const float* offset_matrix, float* norm_log_count_mat, int cells, int genes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements =  cells*genes;

    if (idx < total_elements) {
        int row = idx / cells;
        int col = idx % cells;
        norm_log_count_mat[idx] = log1p(y[idx] / exp(offset_matrix[col]));
    }
}

// CUDA implementation of init_beta function
void init_beta(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> const &Y_host,
               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> const &design_matrix_host,
               Eigen::VectorXf const & offset_host,
               float* beta, int n, int p, int m, int k) {
  
  std::size_t cells = design_matrix_host.rows();
  std::size_t features = design_matrix_host.cols();
  std::size_t genes = Y_host.rows();
  
  
  // Allocate device memory

  float *Y, *design_matrix, *offset, *norm_log_count_mat,*Q,*R;
  
  cudaMalloc((void **)&Y, genes * cells * sizeof(float));
  cudaMalloc((void **)&design_matrix, features * cells * sizeof(float));
  cudaMalloc((void **)&offset, cells * sizeof(float));
  //scoprire la size guardando i conti
  cudaMalloc((void **)&norm_log_count_mat, genes * cells * sizeof(float));

  cudaMalloc(&Q, features * cells * sizeof(float));
  cudaMalloc(&R, features * features * sizeof(float));
  cudaMalloc(&beta, genes * features * sizeof(float));

  // Copy data to device, use toGPU

  toGPU(Y_host, Y);
  toGPU(design_matrix_host, design_matrix);
  toGPU(offset_host, offset);
  
  //cudaMemcpy(d_y, y, m * k * sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_design_matrix, design_matrix, n * p * sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_offset_matrix, offset_matrix, m * k * sizeof(double), cudaMemcpyHostToDevice);

    // Step 1: Compute log1p(y / exp(offset_matrix))
    int total_elements = cells * genes ;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    compute_log1p<<<gridSize, blockSize>>>(Y, offset, norm_log_count_mat,cells, genes);
    cudaDeviceSynchronize();

    // Step 2: Perform QR decomposition on design_matrix
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Workspace and info
    float *d_work;
    int *dev_info;
    int lwork;

    cusolverDnSgeqrf_bufferSize(handle, features, cells, design_matrix, features, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));
    cudaMalloc(&dev_info, sizeof(int));

    cusolverDnSgeqrf(handle, features, cells, design_matrix, features, R, d_work, lwork, dev_info);
    cudaDeviceSynchronize();

    // Extract Q and R
    cusolverDnSormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, cells, genes, features, design_matrix, cells, R, Q, cells, d_work, lwork, dev_info);
    cudaDeviceSynchronize();
    /*
    cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N,
                 cells, genes, features, 
                 design_matrix, cells, R, 
                 Q, cells, d_work, lwork, dev_info);
    */
    // Step 3: Compute Q^T * norm_log_count_mat
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float alpha = 1.0;
    float beta_scalar = 0.0;
    float *QT_norm_log;
    cudaMalloc(&QT_norm_log, features * cells * sizeof(float));
       // n: number of rows in design_matrix
    // p: number of columns in design_matrix
    // m: number of rows in y
    // k: number of columns in y
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cells, genes, cells, &alpha, Q, cells, norm_log_count_mat, cells, &beta_scalar, QT_norm_log, features);
    /*
      cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
            features, genes, cells, &alpha, 
            Q, cells, norm_log_count_mat, cells, 
            &beta_scalar, QT_norm_log, features);

     */
    // Step 4: Solve R * beta = QT_norm_log using cuBLAS Dtrsm

cublasStrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            features, cells, &alpha,
            R, features, // R matrix (left-hand side)
            QT_norm_log, features); // Q^T * norm_log_count_mat (right-hand side)


    cudaMemcpy(beta, QT_norm_log, features * genes * sizeof(float), cudaMemcpyDeviceToHost);


    cublasDestroy(cublas_handle);
    cusolverDnDestroy(handle);
}
