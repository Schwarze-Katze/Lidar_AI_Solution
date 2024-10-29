import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import os
import struct
import sensor_msgs.point_cloud2 as pc2


# 定义文件路径
file_names = [
    "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
    "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"
]
image_dir = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data"  # 替换为你的图像目录
points_file = "/home/rancho/1lhr/Lidar_AI_Solution/pointpillars_ws/src/bevfusion/example-data/points.tensor"


class ImagePointCloudPublisher:
    def __init__(self):
        rospy.init_node('image_pointcloud_publisher', anonymous=True)

        # 创建发布者
        self.image_pub = rospy.Publisher(
            '/camera/image_raw', Image, queue_size=10)
        self.pointcloud_pub = rospy.Publisher(
            '/lidar/points', PointCloud2, queue_size=10)

        self.bridge = CvBridge()
        self.rate = rospy.Rate(1)

    def load_points(self):
        # 读取点云数据
        points = []
        with open(points_file, 'rb') as f:
            while True:
                # 读取点云数据，假设每个点包含 x, y, z, intensity
                data = f.read(16)  # 每个点占用 4*4 字节
                if len(data) < 16:  # 检查是否读取到足够的字节
                    if data:  # 如果还有数据，但不足16字节，处理异常
                        rospy.logwarn("Warning: Incomplete point data, skipping.")
                    break  # 如果没有数据，结束循环
                point = struct.unpack('ffff', data)  # 假设以 float 格式存储
                points.append(point)
                # print(point)
                
        return points
        # return [(p[0], p[1], p[2]) for p in points]

    def publish_data(self):  # 设置发布频率

        img_rate = rospy.Rate(0.3)
        # 加载点云数据
        points = self.load_points()
        
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        
        fields = [pc2.PointField('x', 0, PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, PointField.FLOAT32, 1),
                  pc2.PointField('i', 12, PointField.FLOAT32, 1)]
        pc2_msg = pc2.create_cloud(header, fields, points)
        self.pointcloud_pub.publish(pc2_msg)
        
        # points = None

        # for file_name in file_names:
        #     # 读取图像
        #     image_path = os.path.join(image_dir, file_name)
        #     cv_image = cv2.imread(image_path)

        #     if cv_image is None:
        #         rospy.logerr(f"Could not read image: {image_path}")
        #         continue

        #     # 转换为 ROS 图像消息
        #     ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        #     self.image_pub.publish(ros_image)

        #     # 发布点云数据
        #     # if points:
        #     #     header = rospy.Header()
        #     #     header.stamp = rospy.Time.now()
        #     #     header.frame_id = 'lidar_frame'  # 根据需要设置帧 ID

        #     #     # 创建 PointCloud2 消息
        #     #     pc2_msg = pc2.create_cloud_xyz32(header, points)
        #     #     self.pointcloud_pub.publish(pc2_msg)

        #     img_rate.sleep()
        print("pub")
        self.rate.sleep()


if __name__ == '__main__':
    try:
        publisher = ImagePointCloudPublisher()
        while not rospy.is_shutdown():
            publisher.publish_data()
    except rospy.ROSInterruptException:
        pass
