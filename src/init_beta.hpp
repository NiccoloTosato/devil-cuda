
void init_beta(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> const &Y_host,
               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> const &design_matrix_host,
               Eigen::VectorXf const & offset_host,
               float* beta, int n, int p, int m, int k);
