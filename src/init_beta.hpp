
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> init_beta_external(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> const &Y_host,
               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> const &design_matrix_host,
								 Eigen::VectorXf const & offset_host,int batch_size);
