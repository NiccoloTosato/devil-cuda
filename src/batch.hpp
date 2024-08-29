#include <Eigen/Dense>
/*
Eigen::MatrixXf beta_fit_gpu_external(Eigen::MatrixXf Y_host, Eigen::MatrixXf
X_host, Eigen::MatrixXf mu_beta_host, Eigen::MatrixXf offset_host,
Eigen::VectorXf k_host, int max_iter, float eps);
*/

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
beta_fit_gpu_external(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const &
        Y_host,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const &
        X_host,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const &
        mu_beta_host,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const &
        offset_host,
    Eigen::VectorXf const & k_host, int max_iter, float eps,int batch_size);
