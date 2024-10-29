#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 图像回调函数
void imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
    // 将 img_msg 转换为 cv::Mat
    cv::Mat concatenated_image = cv_bridge::toCvCopy(img_msg, "rgb8")->image;

    // 检查图像是否有效
    if (concatenated_image.empty()) {
        ROS_ERROR("Received empty image!");
        return;
    }

    // 横向分割图像为 6 张
    int segment_width = concatenated_image.cols / 6;
    std::vector<cv::Mat> segments;

    for (int i = 0; i < 6; ++i) {
        cv::Rect roi(i * segment_width, 0, segment_width, concatenated_image.rows);
        segments.push_back(concatenated_image(roi));
    }

    // 保存每个分割图像
    for (int i = 0; i < segments.size(); ++i) {
        std::string segment_path = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/scripts/segment_" + std::to_string(i) + ".jpg";
        cv::imwrite(segment_path, segments[i]); // 保存分割图像
        ROS_INFO("Saved segment %d to %s", i, segment_path.c_str());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_segmenter"); // 初始化 ROS 节点
    ros::NodeHandle nh; // 创建节点句柄

    // 订阅图像话题
    ros::Subscriber image_sub = nh.subscribe("/pub_image_topic", 1, imageCallback); // 替换 "image_topic" 为实际的图像话题名称

    ROS_INFO("Image segmenter node started, waiting for images...");

    // 进入 ROS 循环
    ros::spin();

    return 0;
}