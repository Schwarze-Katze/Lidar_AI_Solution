import open3d as o3d
import numpy as np
import sys
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.visualization import draw_geometries

thr=0.25

def get_box_vertices(x, y, z, w, l, h, rt):
    # w, l, h are width, length, height of the box
    # rt is the rotation around the z-axis
    # Calculate the rotation matrix
    rt=-rt+np.pi/2
    R = np.array([
        [np.cos(rt), -np.sin(rt), 0],
        [np.sin(rt),  np.cos(rt), 0],
        [0,           0,          1]
    ])

    # Half dimensions
    hw, hl, hh = w / 2.0, l / 2.0, h / 2.0

    # 8 corners
    corners = np.array([
        [hl,  hw,  hh],
        [hl, -hw,  hh],
        [-hl, -hw,  hh],
        [-hl,  hw,  hh],
        [hl,  hw, -hh],
        [hl, -hw, -hh],
        [-hl, -hw, -hh],
        [-hl,  hw, -hh]
    ])

    # Rotate and translate corners
    corners = np.dot(corners, R.T) + np.array([x, y, z])

    return corners

# Now, read the data from file and compute vertices for each box


def read_boxes_and_compute_vertices(file_name):
    boxes = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = list(map(float, line.split()))
            x, y, z, w, l, h, vx, vy, rt, box_id, score = parts
            if score < thr:
                continue
            vertices = get_box_vertices(x, y, z, w, l, h, rt)
            boxes.append(np.array(vertices))

    return boxes


def main(pcd,boxes):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="vis")
    # 设置点云大小
    vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    # 设置点的颜色为白色
    pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)
    
    for box in boxes:
        # 指明哪两个顶点之间相连
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                            [0, 4], [1, 5], [2, 6], [3, 7]])
        # 设置点与点之间线段的颜色
        colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
        # 创建Bbox候选框对象
        line_set = o3d.geometry.LineSet()
        # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        # 设置每条线段的颜色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # 把八个顶点的空间信息转换成o3d可以使用的数据类型
        line_set.points = o3d.utility.Vector3dVector(box)
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    assert len(sys.argv) > 1
    codename = sys.argv[1]
    predname = f"prediction/{codename}.txt"
    pcdname = f"test/{codename}.bin"
    boxes = read_boxes_and_compute_vertices(predname)
    
    # lidar_path换成自己的.bin文件路径
    pc = np.fromfile(pcdname, dtype=np.float32, count=-1).reshape([-1, 5])
    # print(pc[0:30])
    # raise
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(pc[:, 0:3].reshape(-1, 3))
    main(point_cloud,boxes)


