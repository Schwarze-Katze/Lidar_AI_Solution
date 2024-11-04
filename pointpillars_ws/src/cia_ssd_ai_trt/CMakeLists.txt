# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0.2)
project(cia)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cuda_pointpillars
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

find_package(CUDA REQUIRED)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")
add_compile_options(-W)
add_compile_options(-std=c++11)


set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/aarch64-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
    -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
    -Xlinker -L/usr/lib/aarch64-linux-gnu/
    -Xlinker -L/usr/aarch64-linux-gnu/lib/
)


# set(CUDA_NVCC_FLAGS -G;-g)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${catkin_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/opt/TensorRT-8.6.1.6/include)
link_directories(/opt/TensorRT-8.6.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(voxelGeneratorlayer SHARED ${PROJECT_SOURCE_DIR}/src/voxelGenerator.cu)
target_link_libraries(voxelGeneratorlayer nvinfer cudart)

cuda_add_library(submConv3dlayer SHARED ${PROJECT_SOURCE_DIR}/src/submConv3dlayer.cu)
target_link_libraries(submConv3dlayer nvinfer cudart)

cuda_add_library(sparseConv3dlayer SHARED ${PROJECT_SOURCE_DIR}/src/sparseConv3dlayer.cu)
target_link_libraries(sparseConv3dlayer nvinfer cudart)

cuda_add_library(sparse2Denselayer SHARED ${PROJECT_SOURCE_DIR}/src/sparse2Dense.cu)
target_link_libraries(sparse2Denselayer nvinfer cudart)

cuda_add_library(zeroPad2dlayer SHARED ${PROJECT_SOURCE_DIR}/src/zeroPad2d.cu)
target_link_libraries(zeroPad2dlayer nvinfer cudart)

cuda_add_library(generateAnchorDecodelayer SHARED ${PROJECT_SOURCE_DIR}/src/generateAnchorDecode.cu)
target_link_libraries(generateAnchorDecodelayer nvinfer cudart)

cuda_add_library(filterBoxByScorelayer SHARED ${PROJECT_SOURCE_DIR}/src/filterBoxByScore.cu)
target_link_libraries(filterBoxByScorelayer nvinfer cudart)

add_executable(cia-ssd-ai-trt ${PROJECT_SOURCE_DIR}/src/cia-ssd-ai-trt.cpp)
add_executable(lidar_cia ${PROJECT_SOURCE_DIR}/src/lidar_cia.cpp)

target_link_libraries(cia-ssd-ai-trt 
  nvinfer
  cudart
  voxelGeneratorlayer
  submConv3dlayer
  sparseConv3dlayer
  sparse2Denselayer
  zeroPad2dlayer
  generateAnchorDecodelayer
  filterBoxByScorelayer
  ${catkin_LIBRARIES}
)

target_link_libraries(lidar_cia
  nvinfer
  cudart
  voxelGeneratorlayer
  submConv3dlayer
  sparseConv3dlayer
  sparse2Denselayer
  zeroPad2dlayer
  generateAnchorDecodelayer
  filterBoxByScorelayer
  ${catkin_LIBRARIES}
)

add_definitions(-O2 -pthread)

