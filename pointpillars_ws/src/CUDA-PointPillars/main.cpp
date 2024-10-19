/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <ros/ros.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include "cuda_runtime.h"




#include "./params.h"
#include "./pointpillar.h"
#define USE_ROS_PCD_INPUT
#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

std::string Src_Path = "";
std::string Data_File = "/data/";
std::string Save_Dir = "/eval/kitti/object/pred_velo/";
std::string Model_File = "/model/pointpillar.onnx";

std::vector<std::string> box_type;

typedef std::unique_lock<std::mutex> ULK;
std::mutex pcd_mtx;
sensor_msgs::PointCloud2 pcd_buf;

void PointCloudCallback(const sensor_msgs::PointCloud2& msg) {
  ULK ulk(pcd_mtx);
  pcd_buf = msg;
#if 0
  // 计算点的数量
  size_t point_count = msg.width * msg.height;

  // 分配内存，用于存储点云数据
  points_data = new float[point_count * 4]; // 每个点对应4个float (x, y, z, intensity)

  // 使用PointCloud2的迭代器来访问数据
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(msg, "intensity");

  for (size_t i = 0; i < point_count; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
    points_data[i * 4 + 0] = *iter_x;       // x
    points_data[i * 4 + 1] = *iter_y;       // y
    points_data[i * 4 + 2] = *iter_z;       // z
    points_data[i * 4 + 3] = *iter_intensity; // intensity
  }
  // 释放分配的内存
  delete[] points_data;
#endif
}

void Getinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
    exit(EXIT_FAILURE);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
{
  std::ofstream ofs;
  ofs.open(file_name, std::ios::out);
  if (ofs.is_open()) {
    for (const auto box : boxes) {
      ofs << box.x << " ";
      ofs << box.y << " ";
      ofs << box.z << " ";
      ofs << box.w << " ";
      ofs << box.l << " ";
      ofs << box.h << " ";
      ofs << box.rt << " ";
      ofs << box.id << " ";
      ofs << box.score << " ";
      ofs << "\n";
    }
  }
  else {
    std::cerr << "Output file cannot be opened!" << std::endl;
  }
  ofs.close();
  std::cout << "Saved prediction in: " << file_name << std::endl;
  return;
};

void PublishBoxPred(std::vector<Bndbox> boxes, ros::Publisher& marker_pub, std::string &color) {
  visualization_msgs::MarkerArray marker_array;
  std::string color_temp;
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto& box = boxes[i];
    const float threshold = 0.3;
    if (box.score < threshold and box_type[box.id] == "car")
    {
      // continue;
      color_temp = "gray";
    }
    else
      color_temp = color;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "rslidar";  // 使用合适的坐标系框架名称
    marker.header.stamp = ros::Time::now();
    marker.ns = "bounding_boxes";
    marker.id = 2 * i + 1;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    // 设置位置信息
    marker.pose.position.x = box.x;
    marker.pose.position.y = box.y;
    marker.pose.position.z = box.z;
    marker.pose.orientation = tf::createQuaternionMsgFromYaw(box.rt + 1.57);

    // 设置尺寸信息
    marker.scale.x = box.l;
    marker.scale.y = box.w;
    marker.scale.z = box.h;
    // 设置颜色
    if (color_temp == "red") {
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.4f;
    }
    else if (color_temp == "green") {
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.4f;
    }
    else if (color_temp == "blue") {
    marker.color.r = 0.0f;
    marker.color.g = 0.0f;
    marker.color.b = 1.0f;
    marker.color.a = 0.4f;
    }
    else {
    marker.color.r = 0.5f;
    marker.color.g = 0.5f;
    marker.color.b = 0.5f;
    marker.color.a = 0.4f;
    }

    marker.lifetime = ros::Duration(0.2);

    // 将标记添加到数组
    marker_array.markers.push_back(marker);


    visualization_msgs::Marker marker_id;
    marker_id = marker;
    marker_id.header.frame_id = "rslidar";  // 使用合适的坐标系框架名称
    marker_id.header.stamp = ros::Time::now();
    marker_id.ns = "bounding_boxes_id";
    marker_id.id = 2 * i;
    marker_id.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker_id.action = visualization_msgs::Marker::ADD;
    if (box.id > box_type.size())
      marker_id.text = "";
    else
      {
        
      marker_id.text = box_type[box.id];
      marker_id.text += "\n";
      marker_id.text += std::to_string(box.score);
      }
    marker_id.pose.position.z += 0.3;
    marker_id.scale.z = 1.0;
    marker_id.color.r = 1.0f;
    marker_id.color.g = 1.0f;
    marker_id.color.b = 1.0f;
    marker_id.color.a = 0.4f;
    marker_array.markers.push_back(marker_id);


  }

  // 发布标记数组
  marker_pub.publish(marker_array);

  ROS_INFO("Published %lu bounding boxes", boxes.size());
}

int main(int argc, char **argv)
{
  Getinfo();
  ros::init(argc, argv, "cuda_pointpillars_node");
  ros::NodeHandle nh("~");
  nh.getParam("src_path", Src_Path);
  std::string lidar_topic;
  nh.getParam("lidar_topic", lidar_topic);
  ros::Subscriber pclsub = nh.subscribe(lidar_topic, 10, PointCloudCallback);
  std::string vis_topic;
  nh.getParam("vis_topic", vis_topic);
  ros::Publisher markerpub = nh.advertise<visualization_msgs::MarkerArray>(vis_topic, 10);
  std::string vis_color;
  nh.getParam("vis_color", vis_color);
  ros::Rate rate(10);
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaStream_t stream = NULL;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  Params params_;

  box_type.emplace_back("car");
  box_type.emplace_back("people");
  box_type.emplace_back("people");


  std::vector<Bndbox> nms_pred;
  nms_pred.reserve(100);

  PointPillar pointpillar(Src_Path + Model_File, stream);

  while(ros::ok())
  {
#ifdef USE_ROS_PCD_INPUT
    //for test, output to file
    int n_zero = 6;
    std::string _str = "0";
    std::string index_str = std::string(n_zero - _str.length(), '0') + _str;

    ULK ulk(pcd_mtx);
    // 计算点的数量
    size_t points_size = pcd_buf.width * pcd_buf.height;

    // 分配内存，用于存储点云数据
    float* points = new float[points_size * 4]; // 每个点对应4个float (x, y, z, intensity)

    if (pcd_buf.fields.empty()) {
      ulk.unlock();
      ROS_INFO("waiting for pointcloud");
      ros::spinOnce();
      rate.sleep();
      continue;
    }
    
    // 使用PointCloud2的迭代器来访问数据
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(pcd_buf, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(pcd_buf, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(pcd_buf, "z");
    sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(pcd_buf, "intensity");

    for (size_t i = 0; i < points_size; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
      points[i * 4 + 0] = *iter_x;       // x
      points[i * 4 + 1] = *iter_y;       // y
      points[i * 4 + 2] = *iter_z;       // z
      points[i * 4 + 3] = *iter_intensity; // intensity
    }
    ulk.unlock();
#else
    std::string dataFile = Src_Path + Data_File;
    int n_zero = 6;
    std::string _str = "0";
    std::string index_str = std::string(n_zero - _str.length(), '0') + _str;
    dataFile += index_str;
    dataFile += ".bin";

    std::cout << "<<<<<<<<<<<" << std::endl;
    std::cout << "load file: " << dataFile << std::endl;

    //load points cloud
    unsigned int length = 0;
    void* data = NULL;
    std::shared_ptr<char> buffer((char*) data, std::default_delete<char[]>());
    loadData(dataFile.data(), &data, &length);
    buffer.reset((char*) data);

    float* points = (float*) buffer.get();
    size_t points_size = length / sizeof(float) / 4;
#endif
    std::cout << "find points num: " << points_size << std::endl;

    float *points_data = nullptr;
    unsigned int points_data_size = points_size * 4 * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
    checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());
    delete[] points;

    cudaEventRecord(start, stream);

    pointpillar.doinfer(points_data, points_size, nms_pred);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

    checkCudaErrors(cudaFree(points_data));

    std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;
    std::string save_file_name = Src_Path + Save_Dir + index_str + ".txt";
#ifdef USE_ROS_PCD_INPUT
    PublishBoxPred(nms_pred, markerpub, vis_color);
#else
    SaveBoxPred(nms_pred, save_file_name);
#endif

    nms_pred.clear();

    std::cout << ">>>>>>>>>>>" << std::endl;
    ros::spinOnce();
    rate.sleep();
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));

  return 0;
}
