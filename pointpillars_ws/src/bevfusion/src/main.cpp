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

#include <cuda_runtime.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"


#include <ros/ros.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>


typedef std::unique_lock<std::mutex> ULK;
std::mutex mtx;
sensor_msgs::PointCloud2 pcd_buf;
// cv_bridge::CvImagePtr cv_ptr;
std::vector<cv::Mat> segments;
bool image_flag;
bool pcd_flag;
nv::Tensor pcd_tensor;

cudaStream_t stream;

typedef unsigned short ushort;//占用2个字节
typedef unsigned int uint;    //占用4个字节

uint as_uint(const float x) {
  return *(uint*) &x;
}
float as_float(const uint x) {
  return *(float*) &x;
}
float half_to_float(const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint e = (x & 0x7C00) >> 10; // exponent
  const uint m = (x & 0x03FF) << 13; // mantissa
  const uint v = as_uint((float) m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
  return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) | ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}

ushort float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
  const uint e = (b & 0x7F800000) >> 23; // exponent
  const uint m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) | ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}


static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                              "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[200];
    sprintf(path, "%s/%s", root.c_str(), file_names[i]);

    int width, height, channels;
    images.push_back(stbi_load(path, &width, &height, &channels, 0));
    printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

nv::Tensor visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
                      cudaStream_t stream) {
  std::vector<nv::Prediction> predictions(bboxes.size());
  memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

  int padding = 300;
  int lidar_size = 1024;
  int content_width = lidar_size + padding * 3;
  int content_height = 1080;
  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f;
  bev_artist_param.image_stride = scene_artist_param.stride;

  auto points = lidar_points.to_device();
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
  bev_visualizer->draw_prediction(predictions, false);
  bev_visualizer->draw_ego();
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
  memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
         sizeof(float) * image_artist_param.viewport_nx4x4.size());

  int gap = 0;
  int camera_width = 500;
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
  int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    int ox = offset_cameras[icamera][0] + content_width / 2;
    int oy = offset_cameras[icamera][1] + content_height / 2;
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(icamera, predictions, xflip);

    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                   device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                     device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  printf("Save to %s\n", save_path.c_str());
  stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
    scene_device_image.to_host(stream).ptr(), 100);
  return scene_device_image;
}

std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {

  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
  voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
  voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = 10;
  voxelization.max_points = 300000;
  voxelization.max_voxels = 160000;
  voxelization.num_feature = 5;

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  // scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
  scn.model = nv::format("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/model/%s/lidar.backbone.xyz.onnx", model.c_str());
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

  if (precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
  } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
  }

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  geometry.geometry_dim = nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = {-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
  transbbox.post_center_range_end = {61.2, 61.2, 10.0};
  transbbox.voxel_size = {0.075, 0.075};
  // transbbox.model = nv::format("model/%s/build/head.bbox.plan", model.c_str());
  transbbox.model = nv::format("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/model/%s/build/head.bbox.plan", model.c_str());

  // if you got an inaccurate boundingbox result please turn on the layernormplugin plan.
  // transbbox.model = nv::format("model/%s/build/head.bbox.layernormplugin.plan", model.c_str());
  transbbox.confidence_threshold = 0.12f;
  transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  // param.camera_model = nv::format("model/%s/build/camera.backbone.plan", model.c_str());
  param.camera_model = nv::format("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/model/%s/build/camera.backbone.plan", model.c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  // param.transfusion = nv::format("model/%s/build/fuser.plan", model.c_str());
  param.transfusion = nv::format("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/model/%s/build/fuser.plan", model.c_str());
  param.transbbox = transbbox;
  // param.camera_vtransform = nv::format("model/%s/build/camera.vtransform.plan", model.c_str());
  param.camera_vtransform = nv::format("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/model/%s/build/camera.vtransform.plan", model.c_str());
  return bevfusion::create_core(param);
}

static std::vector<unsigned char*> convertMatToUnsignedCharPtr(const std::vector<cv::Mat>& segments) {
  std::vector<unsigned char*> ucharPointers;

  for (const auto& mat : segments) {
    if (mat.empty()) {
      std::cerr << "Warning: Empty cv::Mat detected, skipping." << std::endl;
      continue;
    }

    // 确保以 BGR 格式存储数据
    if (mat.type() == CV_8UC3) {
      int rows = mat.rows;
      int cols = mat.cols;
      int channels = mat.channels();
      
      unsigned char* dataPtr = new unsigned char[rows * cols * channels];

      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          for (int ch = 0; ch < channels; ++ch) {
            dataPtr[(r * cols + c) * channels + ch] = mat.at<cv::Vec3b>(r, c)[ch];
          }
        }
      }
      
      // unsigned char* testPtr;
      // std::memcpy(testPtr, mat.data, 900 * 1600 * 3);
      // tt_image.data = mat.data;
      cv::Mat tt_image(900, 1600, CV_8UC3, dataPtr);

      // cv::Mat tt_image = mat;
      
      std::string output_path = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/scripts/test.jpg";
      cv::imwrite(output_path, tt_image);


      ucharPointers.push_back(dataPtr);   // 存储指针
    }
    else {
      std::cerr << "Error: Unsupported cv::Mat type detected." << std::endl;
    }
  }

  return ucharPointers; // 返回指针向量
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if (image_flag) return;
  ROS_WARN("get image");
  // ULK ulk(mtx);
  segments.clear();
  cv::Mat concatenated_image;
  cv::Mat bgr_image;
  try {

    concatenated_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8)->image;

    // std::string output_path = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/build/tt.jpg";
    // cv::imwrite(output_path, concatenated_image);

    cv::cvtColor(concatenated_image, bgr_image, cv::COLOR_RGB2BGR);

  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  int segment_width = concatenated_image.cols / 6;

  for (int i = 0; i < 6; ++i) {
    cv::Rect roi(i * segment_width, 0, segment_width, concatenated_image.rows);
    segments.push_back(bgr_image(roi));

    // std::string segment_path = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/scripts/main_segment_" + std::to_string(i) + ".jpg";
    // cv::imwrite(segment_path, segments[i]); // 保存分割图像

  }

  image_flag = true;
  ROS_WARN("get image ok");
}



void convert_cloud_to_tensor(const pcl::PointCloud<pcl::PointXYZI>& cloud, nv::Tensor& lidar_points) {
  size_t num_points = cloud.size();
  size_t total_size = num_points * 5; // 每个点需要 5 个 float16

  // 创建一个 void* 类型的数组，用于存储转换后的数据
  std::vector<ushort> float16_data(total_size); // n * 2 * 5 需要的 float16 数组

  for (size_t i = 0; i < num_points; ++i) {
    const pcl::PointXYZI& pt = cloud.points[i];
    // std::cout << i << "; " << num_points << std::endl;
    // 将 x, y, z, intensity 转换为 float16 并存储
    float16_data[i * 5 + 0] = float_to_half(pt.x);         // x
    float16_data[i * 5 + 1] = float_to_half(pt.y);         // y
    float16_data[i * 5 + 2] = float_to_half(pt.z);         // z
    float16_data[i * 5 + 3] = float_to_half(pt.intensity);  // intensity
    float16_data[i * 5 + 4] = 0;                         // 补 0
  }

  // 调用 Tensor::from_data 创建张量
  std::vector<int64_t> shape = { static_cast<int64_t>(num_points), 5 };
  lidar_points = nv::Tensor::from_data(static_cast<void*>(float16_data.data()),
    shape, nv::DataType::Float16, false, stream);
}


void PointCloudCallback(const sensor_msgs::PointCloud2& msg) {
  ROS_WARN("get ptc");
  // ULK ulk(mtx);
  // ROS_WARN("get ptc in");
  pcd_buf = msg;
  // std::cout << pcd_buf.header.stamp.toNSec() << std::endl;

  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::fromROSMsg(pcd_buf, cloud);


  // 将数据转换为 nv::Tensor
  convert_cloud_to_tensor(cloud, pcd_tensor);


  pcd_flag = true;
  ROS_WARN("get ptc ok");
}


void PublishBoxPred(std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, ros::Publisher& marker_pub, std::string& color) {
  visualization_msgs::MarkerArray del_array;

  visualization_msgs::Marker new_marker;
  new_marker.action = visualization_msgs::Marker::DELETEALL;

  del_array.markers.push_back(new_marker);
  marker_pub.publish(del_array);


  visualization_msgs::MarkerArray marker_array;
  std::string color_temp;


  // for (size_t i = 0; i < boxes.size(); ++i) {
  //     const auto& box = boxes[i];
  //     std::cout << "!!! " << box.x << ", " << box.y << std::endl;
  //     }

  // return ;

  for (size_t i = 0; i < bboxes.size(); ++i) {

    const auto& box = bboxes[i];
    const float threshold = 0.3;
    // if (box.score < threshold and box_type[box.id] == "car") {
    //   // continue;
    //   color_temp = "gray";
    // }
    // else
    //   color_temp = color;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";  // 使用合适的坐标系框架名称
    marker.header.stamp = ros::Time::now();
    marker.ns = "bounding_boxes";
    marker.id = 2 * i + 1;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    // 设置位置信息
    marker.pose.position.x = box.position.x;
    marker.pose.position.y = box.position.y;
    marker.pose.position.z = box.position.z;


    // std::cout << "!!! " << marker.pose.position << std::endl;
    
    marker.pose.orientation = tf::createQuaternionMsgFromYaw(box.z_rotation);

    // 设置尺寸信息
    marker.scale.x = box.size.l;
    marker.scale.y = box.size.w;
    marker.scale.z = box.size.h;
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

    // marker.lifetime = ros::Duration(0.2);

    // 将标记添加到数组
    marker_array.markers.push_back(marker);


    visualization_msgs::Marker marker_id;
    marker_id = marker;
    marker_id.header.frame_id = "map";  // 使用合适的坐标系框架名称
    marker_id.header.stamp = ros::Time::now();
    marker_id.ns = "bounding_boxes_id";
    marker_id.id = 2 * i;
    marker_id.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker_id.action = visualization_msgs::Marker::ADD;
    // marker_id.text = "1";
    marker_id.text = std::to_string(box.id);
    marker_id.text += "\n";
    marker_id.text += std::to_string(box.score);
    // if (box.id > box_type.size())
    //   marker_id.text = " ";
    // else {

    //   marker_id.text = box_type[box.id];
    //   marker_id.text += "\n";
    //   marker_id.text += std::to_string(box.score);
    // }
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

  ROS_INFO("Published %lu bounding boxes", bboxes.size());
}

void PubTensorAsImage(const nv::Tensor& tensor, ros::Publisher& resimg_pub) {
  // 假设张量的尺寸为 [height, width, channels]
  int height = tensor.size(0);
  int width = tensor.size(1);
  int channels = tensor.size(2);
  std::cout << height << "; " << width << "; " << channels << "; " << std::endl;
  // 检查通道数是否符合 CV_8UC3 格式（即 RGB）
  if (channels != 3 && channels != 1) {
    std::cerr << "Error: Unsupported number of channels: " << channels << std::endl;
    return;
  }

  // 创建 cv::Mat 对象
  cv::Mat image(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1);

  // 获取张量数据指针
  // unsigned char* tensorData = tensor.ptr<unsigned char>();
  auto tensorData = *tensor.data;
  void* void_data = tensorData.data;
  ushort* us_data = static_cast<ushort*>(void_data);
  // std::cout << int(tensorData.dtype) << std::endl;
  // 将张量数据复制到 cv::Mat
  std::memcpy(image.data, us_data, height * width * channels);

  sensor_msgs::ImagePtr img_msg;
  img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  resimg_pub.publish(img_msg);
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "bevfusion");
  ros::NodeHandle nh;
  
  const char* data = "example-data";
  const char* model     = "resnet50int8";
  const char* precision = "int8";
  ros::Rate rate(10);

  std::string lidar_topic;
  // nh.getParam("lidar_topic", lidar_topic);
  lidar_topic = "/lidar_topic";
  ros::Subscriber pclsub = nh.subscribe(lidar_topic, 1, PointCloudCallback);
  std::string image_topic;
  // nh.getParam("image_topic", image_topic);
  image_topic = "/image_topic";
  ros::Subscriber imgsub = nh.subscribe(image_topic, 1, imageCallback);

  std::string resimage_topic = "/res_image_topic";
  ros::Publisher resimg_pub = nh.advertise<sensor_msgs::Image>(resimage_topic, 1);

  
  std::string vis_topic;
  // nh.getParam("vis_topic", vis_topic);
  vis_topic = "boxs";
  ros::Publisher markerpub = nh.advertise<visualization_msgs::MarkerArray>(vis_topic, 10);
  std::string vis_color;
  vis_color = "red";
  // nh.getParam("vis_color", vis_color);


  // std::string root = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data";
  // auto tt_images = load_images(root);
  // cv::Mat tt_image(900, 1600, CV_8UC3);
  // tt_image.data = tt_images[0];

  // cv::Mat ttt_image;
  // cv::cvtColor(tt_image, ttt_image, cv::COLOR_RGB2BGR);
  // std::string output_path = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/scripts/test.jpg";
  // cv::imwrite(output_path, ttt_image);


  image_flag = false;
  pcd_flag = false;

  std::vector<unsigned char*> images;

  
  if (argc > 1) data = argv[1];
  if (argc > 2) model     = argv[2];
  if (argc > 3) precision = argv[3];
  dlopen("libcustom_layernorm.so", RTLD_NOW);

  ROS_WARN("create core");
  auto core = create_core(model, precision);

  if (core == nullptr) {
    printf("Core has been failed.\n");
    return -1;
  }

  ROS_WARN("cudaStreamCreate");
  cudaStreamCreate(&stream);
 
  core->print();
  core->set_timer(true);
  ROS_WARN("cudaStreamCreate ok");

  // Load matrix to host
  // /home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/
  // auto camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data), false);
  // auto camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data), false);
  // auto lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
  // auto img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", data), false);

  auto camera2lidar = nv::Tensor::load("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/camera2lidar.tensor", false);
  auto camera_intrinsics = nv::Tensor::load("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/camera_intrinsics.tensor", false);
  auto lidar2image = nv::Tensor::load("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/lidar2image.tensor", false);
  auto img_aug_matrix = nv::Tensor::load("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/img_aug_matrix.tensor", false);


  core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(),
    stream);
  // core->free_excess_memory();


  ROS_WARN("Load matrix ok");

  while (ros::ok()) {

    ULK ulk(mtx);


    if (!image_flag and !pcd_flag)
    {
      ROS_WARN("flag");
      ros::spinOnce();
      rate.sleep();
      ulk.unlock();
      // image_flag = false;
      // pcd_flag = false;
      continue;
    }
    ROS_WARN("run main");
    auto input_images = convertMatToUnsignedCharPtr(segments);
    
    auto bboxes = core->forward((const unsigned char**) input_images.data(), pcd_tensor.ptr<nvtype::half>(), pcd_tensor.size(0), stream);

    // std::vector<bevfusion::head::transbbox::BoundingBox> bboxes;
    auto res_img = visualize(bboxes, pcd_tensor, input_images, lidar2image, "build/test-cuda-bevfusion.jpg", stream);
    // free_images(images);
    PublishBoxPred(bboxes, markerpub, vis_color);
    // PubTensorAsImage(res_img, resimg_pub);
    
    ulk.unlock();
    image_flag = false;
    pcd_flag = false;
    
    ros::spinOnce();
    rate.sleep();
  }


  

  // // evaluate inference time
  // for (int i = 0; i < 5; ++i) {
  //   core->forward((const unsigned char**)images.data(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
  // }

  // // visualize and save to jpg
  // visualize(bboxes, lidar_points, images, lidar2image, "build/cuda-bevfusion.jpg", stream);

  // // destroy memory
  // free_images(images);
  checkRuntime(cudaStreamDestroy(stream));

  printf("[Warning]: If you got an inaccurate boundingbox result please turn on the layernormplugin plan. (main.cpp:207)\n");
  return 0;
}