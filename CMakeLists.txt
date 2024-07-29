cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(devil-cuda LANGUAGES CXX CUDA)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit REQUIRED)

#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)

add_executable(main.x src/main.cpp)
add_executable(test.x src/test.cpp src/inverse.cpp)


target_link_libraries(main.x CUDA::cudart CUDA::cublas)
target_link_libraries(test.x gtest_main CUDA::cudart CUDA::cublas)

enable_testing()
include(GoogleTest)
gtest_discover_tests(test.x)


