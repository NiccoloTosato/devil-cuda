#include <Eigen/Dense>
/*
Eigen::MatrixXf beta_fit_gpu_external(Eigen::MatrixXf Y_host, Eigen::MatrixXf
X_host, Eigen::MatrixXf mu_beta_host, Eigen::MatrixXf offset_host,
Eigen::VectorXf k_host, int max_iter, float eps);
*/


Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
beta_fit_gpu_external(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Y_host,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        X_host,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        mu_beta_host,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> offset_host,
Eigen::VectorXf k_host, int max_iter, float eps);

void hello();

