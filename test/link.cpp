#include "batch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <chrono> 
#include <cmath>

std::vector<float> readDatFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Unable to open file " << filename << std::endl;
    return {};
  }
  // Get the file size
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  // Read the data
  std::vector<float> data(size / sizeof(float));
  if (file.read(reinterpret_cast<char *>(data.data()), size)) {
    std::cout << "Loading file " << filename << " Success,size: " << data.size() << std::endl;
    return data;
  } else {
    std::cerr << "Error reading file " << filename << std::endl;
    return {};
  }
}

int main() {

 auto X_hostv = readDatFile("../data/X.dat");
 auto Y_hostv = readDatFile("../data/Y.dat");
 auto offset_hostv = readDatFile("../data/off.dat");
 auto mu_beta_hostv = readDatFile("../data/mu_beta.dat");
 auto k_hostv = readDatFile("../data/K.dat");
 auto beta_out_hostv = readDatFile("../data/beta_out.dat");
 for (auto &x : k_hostv)
   x=1/x;
 int cells=1024;
 int genes=64;
 int features=2;

 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>   Y_host(Y_hostv.data(), genes,cells); //
 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  X_host(X_hostv.data(), cells,features); //
 Eigen::Map<
     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
     mu_beta_host(mu_beta_hostv.data(), genes, features);
 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  mu_beta_host_out(beta_out_hostv.data(),genes,features); 
 Eigen::Map<Eigen::VectorXf>  offset_host(offset_hostv.data(), cells);
 Eigen::Map<Eigen::VectorXf>  k_host(k_hostv.data(), genes);
 auto start_time = std::chrono::high_resolution_clock::now();
 auto Y_hostT = Y_host.transpose().eval();
 auto X_hostT = X_host.transpose().eval();
 auto mu_beta_hostT = mu_beta_host.transpose().eval();
 auto end_time = std::chrono::high_resolution_clock::now();
 auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Time taken for parallel transpose: " << duration << " microseconds" << std::endl;

 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y = Y_hostT;
 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> X = X_hostT;
 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mu_beta =  mu_beta_hostT;


 std::cout << "Data loaded" << std::endl;
 std::vector<int> iterations(genes);
 auto result = beta_fit_gpu_external(Y, X, mu_beta, offset_host, k_host, 700, 1e-8,32, iterations);
  std::cout << "DONE \n";
  // std::cout << result.transpose() << std::endl;
  // std::cout << "Mu beta\n" << mu_beta_host_out << std::endl;
  auto mean_diff = 0.0;
  auto count_diff=0;
  for (int i = 0; i < result.size(); ++i) {
    auto diff=(abs((result.data()[i] - mu_beta_host_out.data()[i]) / abs(mu_beta_host_out.data()[i])) *
	       100);
   if (diff > 25) {
     std::cout << "Diff: " << diff << " %, \tResults: " << result.data()[i]
               << " Correct:" << mu_beta_host_out.data()[i] << std::endl;
     count_diff++;
   } else {
     std::cout << "Diff: " << diff << " %" << std::endl;
   mean_diff+=diff;
   }

  }
  std::cout << "Difform: " << count_diff << "\nMean diff: " << mean_diff/(result.size()-count_diff) << std::endl;
 for (int i : iterations) {
   std::cout << " " << i ;
 }
 std::cout << std::endl;

}
