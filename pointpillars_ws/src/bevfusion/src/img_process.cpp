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

ros::Publisher img_pub;

cv::Mat concatenate_images(const cv::Mat& image, const cv::Mat& black) {
    cv::Mat result;
    std::vector<cv::Mat> images;
    images.clear();
    images.emplace_back(image);
    images.emplace_back(black);
    images.emplace_back(black);
    images.emplace_back(black);
    images.emplace_back(black);
    images.emplace_back(black);
    cv::hconcat(images, result);  // 横向拼接
    return result;
}

void imageCallback(const sensor_msgs::CompressedImagePtr& img_msg) {
    // 将 img_msg 转换为 cv::Mat
    cv::Mat sub_image = cv_bridge::toCvCopy(img_msg, "rgb8")->image;
    cv::Size img_size = cv::Size(1600, 900);
    cv::Mat new_image;
    int original_width = sub_image.cols;
    int original_height = sub_image.rows;
    
    int new_height = static_cast<int>(original_width * 9 / 16);
    int y = (original_height - new_height) / 2;
    cv::Rect roi(0, y, original_width, new_height);
    cv::Mat cropped_image = sub_image(roi);


    cv::resize(cropped_image, new_image, img_size, 0, 0, cv::INTER_AREA);
    cv::Mat black = cv::Mat(img_size, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // 检查图像是否有效
    if (sub_image.empty()) {
        ROS_ERROR("Received empty image!");
        return;
    }

    auto concatenated_image = concatenate_images(new_image, black);


    cv::Mat bgr_image;
    cv::cvtColor(concatenated_image, bgr_image, cv::COLOR_RGB2BGR);

    auto new_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", bgr_image).toImageMsg();

    img_pub.publish(new_img_msg);
    // ROS_WARN("sss");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;
    ros::Rate rate(10);
    std::string image_topic = "/image_topic";
    nh.getParam("pub_image_topic", image_topic);
    img_pub = nh.advertise<sensor_msgs::Image>(image_topic, 1);
    std::string sub_image_topic = "/sub_image_topic";
    nh.getParam("sub_image_topic", sub_image_topic);
    ros::Subscriber image_sub = nh.subscribe(sub_image_topic, 1, imageCallback);
    

    while (ros::ok()) {
        ros::spinOnce();

        rate.sleep();
        ROS_INFO("Pub");
    }


    return 0;
}