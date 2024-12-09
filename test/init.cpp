#include "batch.hpp"
#include "init_beta.hpp"
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

 auto X_hostv = readDatFile("../QR/X.dat");
 auto Y_hostv = readDatFile("../QR/Y.dat");
 auto offset_hostv = readDatFile("../QR/off.dat");
 auto output_hostv = readDatFile("../QR/output.dat");

 int cells=1024;
 int genes=64;
 int features=2;

 Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>  Y_host(Y_hostv.data(), genes,cells); //
 Eigen::Map<
     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
     X_host(X_hostv.data(), cells, features); //
 Eigen::Map<
     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
     output_host(output_hostv.data(), genes, features); //
 
 Eigen::Map<Eigen::VectorXf>  offset_host(offset_hostv.data(), cells);
 std::cout << "Data loaded" << std::endl;
 std::vector<int> iterations(genes);
auto Y_host_t=Y_host.transpose().eval();
auto beta = init_beta_external(Y_host_t,X_host,offset_host);
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> test = beta;

    std::cout << "Beta matrix gpu (size: " << features << " x " << genes << "):" << std::endl;
//std::cout << beta << std::endl;
    for (int i = 0; i < features*genes; ++i) {      // Iterate over rows (features)
	  std::cout << beta.data()[i] << " " << output_hostv.data()[i] << std::endl;
     }

  std::cout << "DONE \n";


}
