cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(devil-cuda LANGUAGES CXX CUDA)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set nvcc as cuda compiler
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_ARCHITECTURES "72")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit REQUIRED)


# Include directories
include_directories(~/.local/include/cutensor ~/.local/include/)
# Link directories
link_directories( ~/.local/lib/cutensor)

#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)

set_source_files_properties(src/inverse.cpp PROPERTIES LANGUAGE CUDA)

add_executable(main.x src/main.cpp src/inverse.cpp)
add_executable(test.x src/test.cpp src/inverse.cpp src/einsum.cpp)

target_link_libraries(main.x CUDA::cudart CUDA::cublas CUDA::cusolver)
target_link_libraries(test.x gtest_main CUDA::cudart CUDA::cusolver cutensor)

enable_testing()
include(GoogleTest)
gtest_discover_tests(test.x)


