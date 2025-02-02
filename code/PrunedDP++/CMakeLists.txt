cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
message("generator:${CMAKE_GENERATOR}")
message("C++ compiler:${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3")
set(CMAKE_CXX_STANDARD 17)
# * Determine GPU architectures
# * Enable the CMake CUDA language
project(PrunedDP++ VERSION 0.0.1 LANGUAGES CXX)

find_package(Boost REQUIRED)
message("${Boost_INCLUDE_DIRS}")
message("${Boost_LIBRARIES}")
# 包含Boost头文件目录
include_directories(${Boost_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/main.cpp)
include_directories(include )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
aux_source_directory(src DIR_SRCS)
 