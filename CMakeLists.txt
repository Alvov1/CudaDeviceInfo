cmake_minimum_required(VERSION 3.28)
project(CudaDeviceInfo CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaDeviceInfo main.cu)

set_target_properties(CudaDeviceInfo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
