#include "utils.hpp"
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <limits>
#include <math.h>
#include <Eigen/Dense>
#include <optional>


template<typename T>
void toGPU(T vec,float* const vec_gpu) {
  CUDA_CHECK( cudaMemcpy(vec_gpu, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice) );
}

// Kernel for element-wise operation log1p(y / exp(offset_matrix))
__global__ void compute_log1p(const float* y, const float* offset_matrix, float* norm_log_count_mat, int cells, int genes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements =  cells * genes;

    if (idx < total_elements) {
        int row = idx / genes;
        int col = idx % genes;
        norm_log_count_mat[idx] = log1p(y[idx] / exp(offset_matrix[row]));
    }
}
// CUDA implementation of init_beta function
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor> init_beta_external(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,       Eigen::ColMajor> const &Y_host,
               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor> const &design_matrix_host,
               Eigen::VectorXf const & offset_host ) {

    // Adjusted dimensions
    std::size_t cells = design_matrix_host.cols();    // m_rows
    std::size_t features = design_matrix_host.rows(); // n_cols
    std::size_t genes = Y_host.cols();

    int m_rows = static_cast<int>(cells);    // Number of rows (cells)
    int n_cols = static_cast<int>(features); // Number of columns (features)
    int min_mn = (m_rows < n_cols) ? m_rows : n_cols;

    // Allocate device memory
    float *Y, *design_matrix, *offset, *norm_log_count_mat;
    CUDA_CHECK(cudaMalloc((void **)&Y, m_rows * genes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&design_matrix, m_rows * n_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&offset, m_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&norm_log_count_mat, m_rows * genes * sizeof(float)));
    //CUDA_CHECK(cudaMalloc(&beta, n_cols * genes * sizeof(float)));

    // Copy data to device
    toGPU(Y_host, Y);
    toGPU(design_matrix_host, design_matrix);
    toGPU(offset_host, offset);

    // Step 1: Compute log1p(y / exp(offset_matrix))
    int total_elements = m_rows * genes;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    compute_log1p<<<gridSize, blockSize>>>(Y, offset, norm_log_count_mat, m_rows, genes);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Perform QR decomposition on design_matrix
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // Allocate tau
    float *tau;
    CUDA_CHECK(cudaMalloc((void **)&tau, min_mn * sizeof(float)));

    // Workspace and info
    int lwork_geqrf = 0;
    float *d_work_geqrf = NULL;
    int *dev_info = NULL;
    CUDA_CHECK(cudaMalloc((void**)&dev_info, sizeof(int)));

    // Get buffer size for geqrf
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
        cusolver_handle, m_rows, n_cols, design_matrix, m_rows, &lwork_geqrf));

    // Allocate workspace for geqrf
    CUDA_CHECK(cudaMalloc((void**)&d_work_geqrf, lwork_geqrf * sizeof(float)));

    // Perform QR factorization
    CUSOLVER_CHECK(cusolverDnSgeqrf(
        cusolver_handle, m_rows, n_cols, design_matrix, m_rows, tau,
        d_work_geqrf, lwork_geqrf, dev_info));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check dev_info for successful QR factorization
    int h_dev_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_dev_info != 0) {
        std::cerr << "QR factorization failed with info = " << h_dev_info << std::endl;

    }

    // Step 3: Apply Q^T to norm_log_count_mat using cusolverDnSormqr
    // Result will overwrite norm_log_count_mat
    float *C; // Result of Q^T * norm_log_count_mat
    CUDA_CHECK(cudaMalloc((void**)&C, m_rows * genes * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(C, norm_log_count_mat, m_rows * genes * sizeof(float), cudaMemcpyDeviceToDevice));

    // Workspace for ormqr
    int lwork_ormqr = 0;
    float *d_work_ormqr = NULL;

    CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(
        cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m_rows, genes, min_mn, design_matrix, m_rows, tau,
        C, m_rows, &lwork_ormqr));

    // Allocate workspace for ormqr
    CUDA_CHECK(cudaMalloc((void**)&d_work_ormqr, lwork_ormqr * sizeof(float)));

    // Apply Q^T to norm_log_count_mat
    CUSOLVER_CHECK(cusolverDnSormqr(
        cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m_rows, genes, min_mn, design_matrix, m_rows, tau,
        C, m_rows, d_work_ormqr, lwork_ormqr, dev_info));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check dev_info for successful application of Q^T
    CUDA_CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_dev_info != 0) {
        std::cerr << "Applying Q^T failed with info = " << h_dev_info << std::endl;

    }

    // Now, C contains Q^T * norm_log_count_mat (m_rows x genes)
    // But we only need the first n_cols rows (since Q^T * norm_log_count_mat results in m_rows x genes matrix)
    // So we can treat C as having leading dimension m_rows but use only n_cols leading rows

    // Step 4: Solve R * beta = C (only first n_cols rows of C)
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    float alpha = 1.0f;

    // R is stored in the upper triangular part of design_matrix (n_cols x n_cols)
    // C (first n_cols rows) is (n_cols x genes)
    CUBLAS_CHECK(cublasStrsm(
        cublas_handle,
        CUBLAS_SIDE_LEFT,          // Solve on the left side: R * X = C
        CUBLAS_FILL_MODE_UPPER,    // R is upper triangular
        CUBLAS_OP_N,               // No transpose on R
        CUBLAS_DIAG_NON_UNIT,      // R has non-unit diagonal
        n_cols,                    // Number of rows of X and R
        genes,                     // Number of columns of X and C
        &alpha,                    // Scalar alpha
        design_matrix,             // Pointer to R within design_matrix
        m_rows,                    // Leading dimension of design_matrix (lda)
        C,                         // Right-hand side matrix C (first n_cols rows)
        m_rows));                  // Leading dimension of C (ldb)

    // Copy the result back to host
    // Since beta is of size n_cols x genes
 
    float* beta_host = (float*)malloc(n_cols * genes * sizeof(float));
if (beta_host == NULL) {
    std::cerr << "Failed to allocate host memory for beta." << std::endl;
    // Handle error appropriately, e.g., exit the program
    exit(EXIT_FAILURE);
}

    CUDA_CHECK(cudaMemcpy2D(beta_host, n_cols * sizeof(float),
                            C, m_rows * sizeof(float),
                            n_cols * sizeof(float), genes,
                            cudaMemcpyDeviceToHost));
Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> beta_matrix(beta_host, n_cols, genes);
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUDA_CHECK(cudaFree(Y));
    CUDA_CHECK(cudaFree(design_matrix));
    CUDA_CHECK(cudaFree(offset));
    CUDA_CHECK(cudaFree(norm_log_count_mat));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(tau));
    CUDA_CHECK(cudaFree(d_work_geqrf));
    CUDA_CHECK(cudaFree(d_work_ormqr));
    CUDA_CHECK(cudaFree(dev_info));
    return beta_matrix;
}
