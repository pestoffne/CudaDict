CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(CudaDict)
SET(CMAKE_CXX_FLAGS "-g -std=c++11")

ADD_EXECUTABLE(CheckCpuTime cpu.cpp check_cpu_time.cpp)

FIND_PACKAGE(CUDA REQUIRED)

CUDA_ADD_EXECUTABLE(CheckGpuTimeWithTest cpu.cpp gpu.h gpu.cu check_gpu_time_with_test.cpp)
