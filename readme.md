# Run
# 工作空间在~/lidar_ws/
## 驱动启动
```
source devel/setup.bash
roslaunch rslidar_sdk start.launch
# 新建终端
roslaunch chcnav rs232_imu.launch # 需要将该文件内COM口号调整为GTIMU接收COM口
```
## PointPillars目标检测
```
source devel/setup.bash
roslaunch cuda_pointpillars start_multiple.launch
```
## CIA-SSD目标检测
```
source devel/setup.bash
roslaunch cia run.launch 
```
## FAST-LIO建图定位
```
source devel/setup.bash
# 以下节点互斥，同时只能启动一个
roslaunch fast_lio mapping_P80.launch 启动机械雷达
roslaunch fast_lio mapping_M1.launch 启动半固态雷达
roslaunch fast_lio mapping_new.launch 启动新研雷达
```
```
cd ~/Downloads/lidar_testbag
rosbag play 2024-09-06-18-15-27.bag
```