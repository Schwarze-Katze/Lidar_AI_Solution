# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 2.8.12)
project(bevfusion)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
set(OpenCV_DIR "/usr/include/opencv4")


find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  sensor_msgs
  std_msgs
  # OpenCV
  cv_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cuda_pointpillars
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

# For cuda-12
# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
# set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
# set(spconv_root ../libraries/3DSparseConvolution/libspconv_cuda12)

# For cuda-11
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")
set(spconv_root libraries/3DSparseConvolution/libspconv_cuda11)

set(CUDASM 89)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_${CUDASM},code=compute_${CUDASM}
)


set(TensorRT_Lib /opt/TensorRT/lib)
set(TensorRT_Inc /opt/TensorRT/include)
set(TensorRT_Bin /opt/TensorRT/bin)
set(CUDA_Lib /usr/local/cuda/lib64)
set(CUDA_Inc /usr/local/cuda/include)
set(CUDA_Bin /usr/local/cuda/bin)
set(CUDA_HOME /usr/local/cuda)

set(CUDNN_Lib /usr/local/cuda/include)


include_directories(
  ${Protobuf_INCLUDE_DIRS}
  ${CUDA_Inc}
  ${TensorRT_Inc}
  ${Python_Inc}
  dependencies/stb
  dependencies/pybind11/include
  libraries/cuOSD/src
  ${spconv_root}/include
  src/common
  src/
)
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})



# set(PATH $TensorRT_Bin:$CUDA_Bin:$PATH)
# set(LD_LIBRARY_PATH $TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH)
# set(PYTHONPATH $BuildDirectory:$PYTHONPATH)


link_directories(
  ${CUDA_Lib}
  ${TensorRT_Lib}
  ${spconv_root}/lib/${arch}
  build
  ${Python_Lib}
)
link_directories(${PCL_LIBRARY_DIRS})



file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/onnx/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  libraries/cuOSD/src/*.cpp
  libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################


####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  src/main.cpp
  ${VISUALIZE_FILES}
)

# add_executable(bevfusion 
  # src/main.cpp
  # ${VISUALIZE_FILES})

target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)

target_link_libraries(bevfusion
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)



cuda_add_executable(pub_msg 
  src/pub_msg.cpp
)

target_link_libraries(pub_msg
  bevfusion_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

cuda_add_executable(sub_msg 
  src/sub_msg.cpp
)

target_link_libraries(sub_msg
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

cuda_add_executable(img_process 
  src/img_process.cpp
)

target_link_libraries(img_process
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

#######################################################################


########################## custom_layernorm.so ################################
cuda_add_library(custom_layernorm SHARED
  src/plugins/custom_layernorm.cu
)

target_link_libraries(custom_layernorm
  libnvinfer.so
  libnvinfer_plugin.so
)
#######################################################################


########################## libpybev.so ################################
# if(${CMAKE_BUILD_PYTHON} STREQUAL "ON")
#   cuda_add_library(pybev SHARED
#     src/python.cpp
#   )

#   target_link_libraries(pybev
#     bevfusion_core
#     libnvinfer.so
#     libspconv.so
#     ${Protobuf_LIBRARIES}
#     libnvinfer_plugin.so
#     $ENV{Python_Soname}
#   )
# endif()
#######################################################################