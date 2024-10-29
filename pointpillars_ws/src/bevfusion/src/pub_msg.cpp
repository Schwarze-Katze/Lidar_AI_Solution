#include <cuda_runtime.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"
#include "stb_image.h"
#include "stb_image_write.h"


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

typedef std::unique_lock<std::mutex> ULK;
std::mutex img_mtx;
std::mutex pcd_mtx;
sensor_msgs::PointCloud2 pcd_buf;

sensor_msgs::ImagePtr img_msg;
cv_bridge::CvImagePtr cv_ptr;

bool image_flag = false;
bool pcd_flag = false;

static std::vector<cv::Mat> load_images(const std::string& root) {
    const char* file_names[] = { "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                                 "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg" };

    std::vector<cv::Mat> images;
    for (int i = 0; i < 6; ++i) {
        std::string path = root + "/" + file_names[i]; // 使用 std::string 处理路径

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR); // 读取图像
        if (!img.empty()) {
            images.push_back(img); // 将读取的图像存储到 vector 中
        }
        else {
            ROS_WARN("Failed to load image: %s", path.c_str());
        }
    }
    return images;
}


cv::Mat concatenate_images(const std::vector<cv::Mat>& images) {
    cv::Mat result;
    cv::hconcat(images, result);  // 横向拼接
    return result;
}




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


void get_ptc(const nv::Tensor& lidar_points, pcl::PointCloud<pcl::PointXYZI>& cloud)
{
    // 假设 lidar_points 是一个 Nx5 的张量，包含 x, y, z, intensity, xx

    for (auto & num: lidar_points.shape)
    {
        std::cout << num << "; ";
    }
    std::cout << std::endl;
    auto data = *lidar_points.data;
    void* float_data = data.data;
    ushort* float16_data = static_cast<ushort*>(float_data);

    for (int i = 0; i < 10; i++) {
        float value = half_to_float(float16_data[i]);
        std::cout << value << "; ";
    }
    
    // std::cout << std::endl;
    // std::size_t length = sizeof(float_data);
    // std::cout << "length:" << length << std::endl;
    // std::cout << float_data[0] << std::endl;

    // std::cout << int(dat.dtype) << "!!!" << std::endl;
    // const float* data = lidar_points.data();
    for (int i = 0; i < lidar_points.shape[0]; ++i) {

        // ROS_WARN("in for");
        pcl::PointXYZI pt;
        pt.x = half_to_float(float16_data[i * lidar_points.shape[1] + 0]);
        pt.y = half_to_float(float16_data[i * lidar_points.shape[1] + 1]);
        pt.z = half_to_float(float16_data[i * lidar_points.shape[1] + 2]);
        pt.intensity = half_to_float(float16_data[i * lidar_points.shape[1] + 3]);
        // std::cout << pt << "; " << i << std::endl;
        cloud.emplace_back(pt);

        // dat = *lidar_points.data;
        // float* point = reinterpret_cast<float*>(lidar_points.data()) + i * 5;  // Each point has 5 floats
        // cloud.points.emplace_back(point[0], point[1], point[2], point[3]); // x, y, z, intensity
    }
}

void publish_point_cloud(ros::Publisher& pcl_pub, const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    sensor_msgs::PointCloud2 cloud_msg;
    

    // 转换为 sensor_msgs::PointCloud2
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";
    pcl_pub.publish(cloud_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;

    ros::Rate rate(10);
    std::string lidar_topic = "/lidar_topic";
    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
    std::string image_topic = "/image_topic";
    ros::Publisher img_pub = nh.advertise<sensor_msgs::Image>(image_topic, 1);

    // 加载点云数据
    auto lidar_points = nv::Tensor::load("/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/points.tensor", false);
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.points.clear();
    get_ptc(lidar_points, cloud);

    
    // 加载图片
    std::string image_root = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/";  // 替换为实际路径
    auto image_data = load_images(image_root);

    // 拼接图片
    cv::Mat concatenated_image = concatenate_images(image_data);

    std::string output_path = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/scripts/concatenated_image.jpg";
    cv::imwrite(output_path, concatenated_image);

    cv::Mat bgr_image;
    cv::cvtColor(concatenated_image, bgr_image, cv::COLOR_RGB2BGR);  // 转换为 BGR 格式

    // 发布图像数据
    // cv_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr_image);
    // img_pub.publish(cv_ptr->toImageMsg());
    img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr_image).toImageMsg();

    

    while (ros::ok()) {
        ros::spinOnce();

        img_pub.publish(img_msg);

        // 发布点云
        publish_point_cloud(pcl_pub, cloud);
        rate.sleep();
        ROS_INFO("Pub");
    }


    return 0;
}