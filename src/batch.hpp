#include <Eigen/Dense>
void beta_fit_gpu(Eigen::MatrixXf Y_host, Eigen::MatrixXf X_host, Eigen::MatrixXf mu_beta_host, Eigen::MatrixXf offset_host, Eigen::VectorXf k_host, int max_iter, float eps);

