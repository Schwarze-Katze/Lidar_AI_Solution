/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <ros/ros.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <memory>
#include <chrono>
#include <dirent.h>

#include "common.h"
#include "centerpoint.h"

#define USE_ROS_PCD_INPUT

std::string Src_Path = "";
std::string Data_File = "/data/";
std::string Save_Dir = "/data/prediction/";
std::string Model_File = "/model/rpn_centerhead_sim.plan";

std::vector<std::string> box_type;

typedef std::unique_lock<std::mutex> ULK;
std::mutex pcd_mtx;
sensor_msgs::PointCloud2 pcd_buf;

void PointCloudCallback(const sensor_msgs::PointCloud2& msg) {
    // ROS_INFO("pointcloud received");
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

void GetDeviceInfo()
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

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int getFolderFile(std::string path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file = ent->d_name;
            if(hasEnding(file, suffix)){
                files.push_back(file.substr(0, file.length()-4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path.c_str());
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

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
    ofs.setf(std::ios::fixed, std::ios::floatfield);
    ofs.precision(5);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.vx << " ";
          ofs << box.vy << " ";
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
}

void PublishBoxPred(std::vector<Bndbox> boxes, ros::Publisher& marker_pub, std::string& color) {
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];





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
        if (color == "red") {
            marker.color.r = 1.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.4f;
        }
        else if (color == "green") {
            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.4f;
        }
        else if (color == "blue") {
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
            marker_id.text = box_type[box.id];
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

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage: \n"
        "    ./centerpoint_infer ../data/test/\n"
        "    Run centerpoint(voxelnet) inference with data under ../data/test/\n"
        "    Optional: --verbose, enable verbose log level\n"
    );
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
    // if (argc < 2)
    //     help();

    // const char *value = nullptr;
    bool verbose = false;
    // for (int i = 2; i < argc; ++i) {
    //     if (startswith(argv[i], "--verbose", &value)) {
    //         verbose = true;
    //     } else {
    //         help();
    //     }
    // }

    GetDeviceInfo();
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

    std::vector<std::string> files;
    getFolderFile(Src_Path + Data_File, files);

    std::cout << "Total " << files.size() << std::endl;

    Params params;
    cudaStream_t stream = NULL;
    checkCudaErrors(cudaStreamCreate(&stream));


    box_type.emplace_back("car");
    box_type.emplace_back("people");
    box_type.emplace_back("people");

    CenterPoint centerpoint(Src_Path + Model_File, verbose);
    centerpoint.prepare();

    while(ros::ok())
    {

#ifdef USE_ROS_PCD_INPUT
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
        std::string dataFile = data_folder + file + ".bin";

        std::cout << "\n<<<<<<<<<<<" <<std::endl;
        std::cout << "load file: "<< dataFile <<std::endl;

        unsigned int length = 0;
        void* points = NULL;

        loadData(dataFile.c_str(), &points, &length);
        size_t points_size = length / (params.feature_num * sizeof(float));
#endif
        std::cout << "find points num: " << points_size << std::endl;

        float* points_data = nullptr;
        unsigned int points_data_size = points_size * 4 * sizeof(float);
        checkCudaErrors(cudaMallocManaged((void**) &points_data, points_data_size));
        checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyHostToDevice));
        ROS_INFO("Start inferring...");

        centerpoint.doinfer(points_data, points_size, stream);
        ROS_INFO("Stop");

#ifdef USE_ROS_PCD_INPUT
        PublishBoxPred(centerpoint.nms_pred_, markerpub, vis_color);
#else
        std::string save_file_name = Src_Path + Save_Dir + file + ".txt";
        SaveBoxPred(nms_pred, save_file_name);
#endif

        std::cout << ">>>>>>>>>>>" << std::endl;
        ros::spinOnce();
        rate.sleep();
    }

    centerpoint.perf_report();
    // checkCudaErrors(cudaFree(points_data));
    checkCudaErrors(cudaStreamDestroy(stream));
    return 0;
}