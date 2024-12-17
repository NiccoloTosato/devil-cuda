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
    int i = idx % cells; // the max
    int j = idx / cells; // the max
    if (idx < total_elements) {
        norm_log_count_mat[idx] = log1pf(y[idx] / expf(offset_matrix[j]));
    }
}
__global__ void norm_log_count_kernel(float* __restrict__ norm_log_count_mat,
                                      const float* __restrict__ offset,
                                      int genes, int cells)
{

    int j = blockIdx.y * blockDim.y + threadIdx.y; // i in [0, N-1],cols
    int i = blockIdx.x * blockDim.x + threadIdx.x; // j in [0, M-1],row
    
if (i < genes && j < cells) {
  //float val = y[i+genes*j];
   float val = norm_log_count_mat[j+cells*i];
        float exp_off = expf(offset[j]);
        float ratio = val / exp_off;
        float out_val = log1pf(ratio);

        norm_log_count_mat[j+cells*i] =out_val;// y[i+genes*j] ;//out_val;
	//norm_log_count_mat[i+genes*j] =out_val;// y[i+genes*j] ;//out_val;
    }
}

__global__ void printMatrixKernel(float* d_matrix, int rows, int cols,int row_limit,int col_limit) {
    // Only one thread (e.g., thread (0,0,0) in block (0,0,0)) should handle the printing to avoid clutter
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < row_limit; i++) {
            printf("Row %d: ", i);
            for (int j = 0; j < col_limit; j++) {
                int idx = j * rows + i;
                printf("%2.2f ", d_matrix[idx]);
            }
            printf("\n");
        }
    }
}

// CUDA implementation of init_beta function
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
	      Eigen::ColMajor> init_beta_external(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,       Eigen::ColMajor> const &Y_host,
						  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
						  Eigen::ColMajor> const &design_matrix_host,
						  Eigen::VectorXf const & offset_host,   int batch_size ) {

  // Adjusted dimensions
  std::size_t cells = design_matrix_host.rows();    // m_rows, OK
  std::size_t features = design_matrix_host.cols(); // n_cols, OK
  std::size_t genes = Y_host.cols();
  int min_mn = (cells < features) ? cells : features;

  // Allocate device memory
  float *Y, *design_matrix, *offset;

  CUDA_CHECK(cudaMalloc((void **)&design_matrix, cells * features * sizeof(float))); //OK
  CUDA_CHECK(cudaMalloc((void **)&offset, cells * sizeof(float))); //OK
  // CUDA_CHECK(cudaMalloc(&beta, n_cols * genes * sizeof(float)));


  // Copy data to device
    

  toGPU(design_matrix_host, design_matrix); //once
  toGPU(offset_host, offset); //once

  //  std::cout<<std::endl;
  //  std::cout<<"offset matrix gpu\n";
  //  printMatrixKernel<<<1, 1>>>(offset, cells, 1, 10, 1);
  //  CUDA_CHECK(cudaDeviceSynchronize());
  //  std::cout<<std::endl;
  // batch


  int num_batch = genes / batch_size;
  //CUDA_CHECK(cudaMalloc((void **)&norm_log_count_mat, cells * batch_size * sizeof(float))); //OK
  CUDA_CHECK(cudaMalloc((void **)&Y, cells * batch_size * sizeof(float))); //OK
  float* beta_host = (float*)malloc(features * genes * sizeof(float));
  if (beta_host == NULL) {
    std::cerr << "Failed to allocate host memory for beta." << std::endl;
    exit(EXIT_FAILURE);
  }
  cusolverDnHandle_t cusolver_handle;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));



  //workspace
  // Allocate tau
  float *tau;
  CUDA_CHECK(cudaMalloc((void **)&tau, min_mn * sizeof(float))); //OK
  // Workspace and info
  int lwork_geqrf = 0;
  float *d_work_geqrf = NULL;
  int *dev_info = NULL;
  CUDA_CHECK(cudaMalloc((void**)&dev_info, sizeof(int)));
  //si e' giusto
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
					     cusolver_handle, cells, features, design_matrix, cells, &lwork_geqrf)); //OK
  // Allocate workspace for geqrf
  CUDA_CHECK(cudaMalloc((void**)&d_work_geqrf, lwork_geqrf * sizeof(float))); //OK
    // Perform QR factorization
    int lwork_ormqr = 0;
    float *d_work_ormqr = NULL;

    CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(
					       cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
					       cells, batch_size, min_mn, design_matrix, cells, tau,
					       Y, cells, &lwork_ormqr));
    CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver_handle, cells, features,
                                      design_matrix, cells, tau, d_work_geqrf,
                                      lwork_geqrf, dev_info)); // OK

  CUDA_CHECK(cudaFree(d_work_geqrf));
    // Allocate workspace for ormqr
    CUDA_CHECK(cudaMalloc((void**)&d_work_ormqr, lwork_ormqr * sizeof(float)));
      for (int i = 0; i < num_batch; i++) {
	float* h_design=(float*)malloc(sizeof(float)*10);
	std::cout<<std::endl;
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK( cudaMemcpy(Y, ((float*) Y_host.data())+batch_size*cells*i, batch_size *cells* sizeof(float), cudaMemcpyHostToDevice) );      
	dim3 blockDim(16, 16);
	dim3 gridDim((batch_size + blockDim.x - 1)/blockDim.x, (cells + blockDim.y - 1)/blockDim.y);
	norm_log_count_kernel<<<gridDim, blockDim>>>(Y, offset,  batch_size, cells);
	CUDA_CHECK(cudaDeviceSynchronize());
	int h_dev_info = 0;
	CUDA_CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
	if (h_dev_info != 0) {
	  std::cerr << "QR factorization failed with info = " << h_dev_info << std::endl;
	}
	// Step 3: Apply Q^T to norm_log_count_mat using cusolverDnSormqr
	// Apply Q^T to norm_log_count_mat
	CUSOLVER_CHECK(cusolverDnSormqr(
					cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
					cells, batch_size, min_mn, design_matrix, cells, tau,
					Y, cells, d_work_ormqr, lwork_ormqr, dev_info));
	CUDA_CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
	if (h_dev_info != 0) {
	  std::cerr << "Applying Q^T failed with info = " << h_dev_info << std::endl;
	}
	// Step 4: Solve R * beta = C (only first features rows of C)
	float alpha = 1.0f;
	CUBLAS_CHECK(cublasStrsm(
				 cublas_handle,
				 CUBLAS_SIDE_LEFT,          // Solve on the left side: R * X = C
				 CUBLAS_FILL_MODE_UPPER,    // R is upper triangular
				 CUBLAS_OP_N,               // No transpose on R
				 CUBLAS_DIAG_NON_UNIT,      // R has non-unit diagonal
				 features,                    // Number of rows of X and R
				 batch_size,                     // Number of columns of X and C
				 &alpha,                    // Scalar alpha
				 design_matrix,             // Pointer to R within design_matrix
				 cells,                    // Leading dimension of design_matrix (lda)
				 Y,                         // Right-hand side matrix C (first n_cols rows)
				 cells));                  // Leading dimension of C (ldb)

	CUDA_CHECK(cudaMemcpy2D(((float *)beta_host) + features * batch_size * i,
				features * sizeof(float),Y,
				cells * sizeof(float), features * sizeof(float),
				batch_size, cudaMemcpyDeviceToHost));
      }

  // Destroy handles
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  // Cleanup
  CUDA_CHECK(cudaFree(Y));
  CUDA_CHECK(cudaFree(design_matrix));
  CUDA_CHECK(cudaFree(offset));
  CUDA_CHECK(cudaFree(tau));

  CUDA_CHECK(cudaFree(d_work_ormqr));
  CUDA_CHECK(cudaFree(dev_info));
  // Copy back the result
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> beta_matrix(beta_host,genes,features);
  return beta_matrix;
}
