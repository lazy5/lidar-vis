# !/usr/bin/env python3
# -- coding: utf-8 --
""" Point cloud data visualization function

Author: fangchenyu
Date: 2022/12/27
"""
from __future__ import print_function

import os
import io
from pathlib import Path
import math
import numpy as np
import cv2
# from PIL import Image

import pypcd

# 可视化区域配置
TOP_Y_MIN = -50   # 主车前进方向左侧距离
TOP_Y_MAX = +50   # 主车前进方向右侧距离
TOP_X_MIN = -30   # 主车前进方向后方距离
TOP_X_MAX = 80    # 主车前进方向前方距离
TOP_Z_MIN = -3    # 主车liadar高度方向下方距离
TOP_Z_MAX = 3     # 主车liadar高度方向上方距离

# 可视化精度配置
TOP_X_DIVISION = 0.1   # 主车左右两侧的可视化精度，单位为米每像素（m/pix）
TOP_Y_DIVISION = 0.1   # 主车前后方向的可视化精度，单位为米每像素（m/pix）
TOP_Z_DIVISION = 0.3   # 主车上下方向的可视化精度，单位为米每像素（m/pix），由于是bev视角，该数据影响不大


def get_scan_from_pcloud(pcloud):
    ''' 从pypcd的数据结构中获取点云数据，生成numpy.array格式
    pcloud: pypcd.PointCloud
    '''
    scan = np.empty((pcloud.points, 4), dtype=np.float32)
    scan[:, 0] = pcloud.pc_data['x']
    scan[:, 1] = pcloud.pc_data['y']
    scan[:, 2] = pcloud.pc_data['z']
    try:
        scan[:, 3] = pcloud.pc_data['intensity']
    except ValueError:
        scan[:, 3] = 255.0
    return scan


def load_pcd(f_pcd):
    ''' 从pcd文件读取点云数据，将数据到数据结构pypcd.PointCloud
    f_pcd: strin, pcd文件路径
    '''
    try:
        if isinstance(f_pcd, str) or isinstance(f_pcd, Path):
            pcloud = pypcd.PointCloud.from_path(f_pcd)
        elif isinstance(f_pcd, bytes):
            # bytesio使用方法链接：https://blog.csdn.net/Victor2code/article/details/105637945
            b_handle = io.BytesIO() # 在内存中创建一个类文件对象，短时效，速度快
            b_handle.write(f_pcd) # 写入数据，同时标志位后移
            b_handle.seek(0) # 将标志位强行拉回到第0字节，重新从开始进行读取
            # f_pcd = BufferedReader(b_handle) # 将类文件对象可以强制转化为文件对象，不过这两种对象的操作方式一致，没有必要转化
            # pcloud = pypcd.PointCloud.from_fileobj(f_pcd)
            pcloud = pypcd.PointCloud.from_fileobj(b_handle)
            b_handle.close() # 回收该对象的内存
        elif isinstance(f_pcd, io.BufferedReader):
            pcloud = pypcd.PointCloud.from_fileobj(f_pcd)
        else:
            raise TypeError(f'load_pcd do not support type {type(f_pcd)}')

    except AssertionError:
        print ("Assertion when load pcd: %s" % f_pcd)
        return None
    scan = get_scan_from_pcloud(pcloud)
    scan[:, 3] /= 255.0
    return scan


def load_bin_scan(velo_filename, dtype=np.float32, n_vec=4):
    ''' 从点云bin文件中获取点云数据，返回numpy.array格式的数据
    velo_filename: strin, bin文件路径
    '''
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def load_semantic_label(label_filename, dtype=np.uint32):
    ''' 从semantic-kitti格式的label文件读取label数据，返回numpy.array格式的数据
    label_filename: string, 语义标签文件路径
    '''
    label = np.fromfile(label_filename, dtype=dtype)
    label = label & 0xFFFF
    return label


def lidar_to_top_coords(x, y):
    ''' 从lidar坐标系下的点(x, y)转化为bev图片坐标(xx, yy)
    '''
    # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
    Xn = int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Yn = int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
    yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)

    return xx, yy


def lidar_to_top(lidar):
    ''' 将点云数据转化为bev下的数据格式
    lidar: np.array, shape is [N, 4], 点云数据，N表示点云数量，4表示描述点的4个维度(x, y, z, intensity)
    top: np.array, shape is [width, hight, channel], 分别表示俯视图片的宽高，channel中主要保存了纵向信息等
    '''
    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze() # 将离散点云的连续坐标量化到规则的网格坐标（x,y）,z轴方向没有量化

    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = Zn - Z0 + 2
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)

    for x in range(Xn):
        # 1. 首先找到网格坐标x方向上的所有点云
        ix = np.where(quantized[:, 0] == x)
        quantized_x = quantized[ix]
        if len(quantized_x) == 0:
            continue
        yy = -x
        for y in range(Yn):
            # 2. 找到网格坐标y方向上的所有点云
            iy = np.where(quantized_x[:, 1] == y)
            quantized_xy = quantized_x[iy]
            count = len(quantized_xy)
            if count == 0:
                continue
            xx = -y
            top[yy, xx, Zn + 1] = min(1, np.log(count + 1) / math.log(32)) # 通道（C=Zn+1）值的依据为该点云柱的点数量
            max_height_point = np.argmax(quantized_xy[:, 2])
            top[yy, xx, Zn] = quantized_xy[max_height_point, 3] # 通道（C=Zn）值的依据为该点云柱最高处点的强度
            for z in range(Zn):
                # 3. 找到网格坐标z方向上的所有点云
                iz = np.where(
                    (quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1)
                )
                quantized_xyz = quantized_xy[iz]
                if len(quantized_xyz) == 0:
                    continue
                zz = z
                # height per slice
                max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                top[yy, xx, zz] = max_height # 遍历到的3d网格的值赋值为网格内最高点在网格内的高度偏差
    return top

def label_lidar_to_top(lidar, NUM_CLASSES=8):
    ''' 将带有语义标签的点云数据转化为bev下的数据格式
    lidar: np.array, shape is [N, 5], 点云数据，N表示点云数量，4表示描述点的4个维度(x, y, z, intensity, class)
    top: np.array, shape is [width, hight, num_class+1], 分别表示俯视图片的宽高，channel中主要保存了点云柱中各个类别的个数以及点的主要类别和最高点处点的类别标签
    '''
    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 4] # 保存点云上每个点的标注类型
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze() # 将离散点云的连续坐标量化到规则的网格坐标（x,y）,z轴方向没有量化

    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = NUM_CLASSES + 2 # 通道表示为统计每个点云柱上各类点的数量（N类），统计每个点云柱上最高处点的类型（1），统计点云柱上最多点的类别（1）
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)

    for x in range(Xn):
        # 1. 首先找到网格坐标x方向上的所有点云
        ix = np.where(quantized[:, 0] == x)
        quantized_x = quantized[ix]
        if len(quantized_x) == 0:
            continue
        yy = -x
        for y in range(Yn):
            # 2. 找到网格坐标y方向上的所有点云
            iy = np.where(quantized_x[:, 1] == y)
            quantized_xy = quantized_x[iy]
            count = len(quantized_xy)
            if count == 0:
                continue
            xx = -y
            for cls_id in range(NUM_CLASSES):
                # top[yy, xx, cls_id] = len(np.where(quantized_xy[:, 3] == cls_id)) # 统计N个通道统计点云柱上各类点的数量
                icls = np.where(quantized_xy[:, 3] == cls_id) # 统计N个通道统计点云柱上各类点的数量
                top[yy, xx, cls_id] = len(quantized_xy[icls])
            max_height_point = np.argmax(quantized_xy[:, 2])
            top[yy, xx, NUM_CLASSES] = quantized_xy[max_height_point, 3] + 1 # 该通道保存的值为该点云柱最高处点的类型
            top[yy, xx, NUM_CLASSES + 1] = np.argmax(top[yy, xx, :NUM_CLASSES]) + 1 # 该通道保存的值为该点云柱数量最多的类别
    return top


def draw_top_image(lidar_top):
    ''' 将从函数lidar_to_top中获得的top数据生成cv2的图片数据，格式为np.array
    lidar_top: np.array, shape is [width, hight, channel], 分别表示俯视图片的宽高，channel中主要保存了纵向信息等
    top_image: np.array, shape is [width, hight, 3]
    '''
    top_image = np.sum(lidar_top, axis=2)
    top_image = top_image - np.min(top_image)
    top_image = (top_image > 0) * 255 # 将灰度图转化为二值图，提升对比度
    # divisor = np.max(top_image) - np.min(top_image)
    # top_image = top_image / divisor * 255
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image


def draw_sementic_aicv_top_image(lidar_top, color_map):
    ''' 将从函数label_lidar_to_top中获得的top数据生成cv2的图片数据，格式为np.array
    lidar_top: np.array, shape is [width, hight, channel], 分别表示俯视图片的宽高，channel中主要保存了纵向信息等
    top_image: np.array, shape is [width, hight, 3], 色彩空间为bgr
    '''
    lidar_top = lidar_top[:, :, -1].astype(np.uint8) # 选取主要类别进行展示（网格呢数量最多的类别）
    color_map = [color_map[i] for i in color_map] # 0表示点云不存在，1表示忽略的点云
    color_map = np.array(color_map).astype("uint8")
    c1 = np.vectorize(color_map[:,0].__getitem__)(lidar_top)
    c2 = np.vectorize(color_map[:,1].__getitem__)(lidar_top)
    c3 = np.vectorize(color_map[:,2].__getitem__)(lidar_top)
    top_image = np.dstack((c3, c2, c1)).astype(np.uint8)
    return top_image


def draw_box3d_on_top(
    image,
    boxes3d,
    color=(255, 255, 255),
    thickness=1,
    scores=None,
    text_lables=None,
    # text_lables=[],
    is_gt=False,
):
    """
        Args:
            boxes3d: (N, 8, 3), N个框8个顶点的xyz坐标
        Returns:
    """
    # if scores is not None and scores.shape[0] >0:
    # print(scores.shape)
    # scores=scores[:,0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        if is_gt:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.line(img, (u0, v0), (u1, v1), (0, 0, 255), thickness, cv2.LINE_AA) # 车头的线标红
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
        # 将label置于框上方
        if text_lables is not None:
            text_pos = (min(u0, u1, u2, u3), min(v0, v1, v2, v3)-5)
            cv2.putText(img, str(text_lables[n]), text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    return img


############ fangchenyu: add else vis component#########################
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = np.array((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ), dtype=float) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(8, axis=1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def show_lidar_topview_with_boxes(pc_velo, objects, objects_pred=None, vis_output_file=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = lidar_to_top(pc_velo)
    top_image = draw_top_image(top_view)
    # print("top_image:", top_image.shape)
    # gt
    gt = boxes_to_corners_3d(objects['bbox3d'])
    lines = objects.get('trackID', None)
    top_image = draw_box3d_on_top(
        # top_image, gt, scores=None, thickness=1, is_gt=True
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        gt = boxes_to_corners_3d(objects_pred['bbox3d'])
        top_image = draw_box3d_on_top(
            top_image, gt, scores=None, thickness=1, is_gt=False
            # top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    if vis_output_file is not None:
        cv2.imwrite(str(vis_output_file), top_image)
    # cv2.imshow("top_image", top_image)
    return top_image


def save_bev_image_by_point_cloud(pc_data, out_path):
    ''' 从lidar点云数据生成bev图片
    pc_data: np.array, shape is [N, 4], 点云数据，N表示点云数量，4表示描述点的4个维度(x, y, z, intensity)
    out_path: string, 输出图片路径
    '''
    top_view = lidar_to_top(pc_data)
    top_image = draw_top_image(top_view)    
    cv2.imwrite(out_path, top_image)
    return top_image


def show_lidar_topview_with_semantic(pc_data, label, color_map, out_path, num_classes=8):
    ''' 从lidar点云数据和语义标签生成bev图片
    pc_data: np.array, shape is [N, 4], 点云数据，N表示点云数量，4表示描述点的4个维度(x, y, z, intensity)
    label: np.array, shape is [N], 每个点的类别
    out_path: string, 输出图片路径
    num_classes: 点云分割的类别数量
    '''
    pc_data = np.concatenate((pc_data, np.expand_dims(label, axis=1)), axis=1)
    top_view = label_lidar_to_top(pc_data, num_classes)
    top_image = draw_sementic_aicv_top_image(top_view, color_map)  
    cv2.imwrite(out_path, top_image)
    return top_image



####################### fangchenyu: use mayavi to visualize pointcloud #########################
import mayavi.mlab as mlab # 利用mayavi库实现点云数据集可视化

def draw_lidar(pc, color=None, bbox3d=None, vis_output_file=None, show_lidar_coord=True, show_fov=False, show_top_view=False):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ or (n,4) of XYZI
        color: numpy array (n) of intensity or label or whatever, the size must same to pc
        bbox3d: numpy array (n,7) of (x, y, z, l, w, h, heading)
        vis_output_file: str, 保存可视化图片的地址
        show_lidar_coord: bool, 是否显示lidar坐标系(True or False)
        show_fov: bool, 是否显示视野范围(True or False)
        show_top_view: bool, 是否为bev视角，True为bev视图，False为跟车视图
    Returns:
        fig: created or used fig
    '''
    if show_top_view:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(600, 1000)) # 建立窗口大小为size的fig，其背景色为黑色
    else:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)) # 建立窗口大小为size的fig，其背景色为黑色
    if color is None: 
        color = pc[:, 0] # 根据x轴的坐标对点云颜色做区分
    else: # 点云中的label为最小值的点与背景色一致，为了做区分，将所有点的label值+1，然后将任意一个点的label值设为0不显示该点
        color += 1
        color[0] = 0
    # draw points 其中colormap指定颜色映射方式，可选方法有{'spectral', 'gnuplot'...}等多种方法，其中’spectral‘方式的映射得到的图像比较好看
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='spectral', scale_factor=1,
                  figure=fig)

    if show_lidar_coord: # 显示lidar坐标系
        # draw origin 画lidar坐标系原点，也就是激光雷达的位置，mode指定形状为球形’sphere‘，颜色为白色
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        # draw axis 画出lidar坐标系，画出3条由原点出发的3d直线
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    if show_fov: # 显示视野范围
        # draw fov (todo: update to real sensor spec.) 画出相机视野范围前方90度
        fov = np.array([  # 45 degree
            [20., 20., 0., 0.],
            [20., -20., 0., 0.],
        ], dtype=np.float64)
        mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1, figure=fig)

    if bbox3d is not None: # 画出3d框
        corners3d = boxes_to_corners_3d(bbox3d)
        fig = draw_corners3d(corners3d, fig)

    # 配置观测相机的位置与角度
    # azimuth: 观测相机位置在xy平面投影与lidar坐标系的x轴的夹角，取值范围(0, 360)，180度为跟车视角
    # elevation: 观测相机位置向量与lidar坐标系z轴的夹角，取值范围为(0,180)，0度为俯视视角。该值接近0或180时，azimuth=0的配置特化传统的xy平面坐标系（L型），azimuth=90为7型
    # focalpoint: 相机焦点配置，不懂，可不配，使用默认值，焦点将位于场景中所有物体的中心
    # distance: 观测相机到焦点（我的理解是lidar坐标系原点）的距离
    if show_top_view:
        mlab.view(azimuth=90, elevation=0, distance=90.0, figure=fig)
    else:
        mlab.view(azimuth=180, elevation=70, distance=62.0, figure=fig)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    if vis_output_file is not None:
        mlab.savefig(vis_output_file, figure=fig)
    # mlab.show() # 交互展示可视化点云数据
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """ 根据3d框的角点坐标，在mayavi的画布中画出3d框
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig
