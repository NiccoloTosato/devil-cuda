int inverseMatrix(cusolverDnHandle_t cusolverH, float *A, float *A_inv,
		  int n);
int inverseMatrix2(cublasHandle_t cublasH, float *A_device[], float *A_inv_device[], int n ,int batchSize);
