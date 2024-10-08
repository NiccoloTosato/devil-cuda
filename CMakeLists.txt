cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(devil-cuda LANGUAGES CXX CUDA)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set nvcc as cuda compiler
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_ARCHITECTURES "80" "70" "72" "75")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp -arch=native -O3")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(~/.local/include/cutensor ~/.local/include/)
# Link directories
link_directories( ~/.local/lib/cutensor)

#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)

set_source_files_properties(src/inverse.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/batch.cpp PROPERTIES LANGUAGE CUDA)


add_executable(main.x src/main.cpp src/kernel.cu src/inverse.cpp)
add_executable(batch.x src/batch.cpp src/kernel.cu src/inverse.cpp)
add_executable(test.x src/test.cpp src/inverse.cpp)

target_link_libraries(main.x CUDA::cudart CUDA::cublas CUDA::cusolver cutensor OpenMP::OpenMP_CXX)
target_link_libraries(batch.x CUDA::cudart CUDA::cublas CUDA::cusolver cutensor OpenMP::OpenMP_CXX)
target_link_libraries(test.x gtest_main CUDA::cudart CUDA::cusolver cutensor)

enable_testing()
include(GoogleTest)
gtest_discover_tests(test.x)


