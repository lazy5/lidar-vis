# !/usr/bin/env python3
# -- coding: utf-8 --
import os
import yaml
import json
import requests
import base64
import numpy as np

from utils import lidar_util


def mkdir(path):
    ''' 生成文件路径 '''
    file_dir = os.path.dirname(path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def get_vis_file(pcd_file, result, save_path, use_mayavi=False):
    ''' 生成可视化结果 '''
    points = lidar_util.load_pcd(pcd_file)
    pred_names, pred_bbox3d, pred_tracks = [], [], []
    for i in range(len(result)):
        pred_names.append(result[i]['type'])
        pred_bbox3d.append((result[i]['bbox3d']))
        pred_tracks.append(i)
    pred_obj = {'names': np.array(pred_names), 'bbox3d': np.array(pred_bbox3d), 'trackID': np.array(pred_tracks)}
    if use_mayavi:
        lidar_util.draw_lidar(points, bbox3d=np.array(pred_bbox3d), vis_output_file=save_path)
    else:
        lidar_util.show_lidar_topview_with_boxes(points, pred_obj, vis_output_file=save_path)


def infer(pcd_file, save_path, vis_mode=True):
    ''' 3d障碍物检测服务推理可视化 '''
    with open(pcd_file, "rb") as f:
        pcd_bytes = f.read()
        base64_data = base64.b64encode(pcd_bytes)
        response = requests.post(url='http://service-url', data={'data': base64_data}, )
        result = response.json()
    if result['retCode'] == 200:
        result = result['retData']['labelData']
        # with open('1.json', 'w') as f:
        #     json.dump(result, f, indent=2)
        # print(result)
        # result_json = json.dumps(result)
        # result = result_json
    else:
        raise Exception(f'Failed to infer for {pcd_file}')
    if vis_mode:
        mkdir(save_path)
        get_vis_file(pcd_file, result['result'], save_path)


def save_bev_img_with_bbox(pcd_file, label_file, save_path, use_mayavi=False, vis_mode=True):
    ''' 从pcd文件保存bev图片 '''
    with open(label_file, 'r') as f:
        result = json.load(f)
    mkdir(save_path)
    get_vis_file(pcd_file, result['result'], save_path, use_mayavi=use_mayavi)



if __name__ == '__main__':
    # 1. 单独的pcd可视化
    pcd_file = 'examples/object-aicv/1.pcd'
    pc_data = lidar_util.load_pcd(pcd_file)
    out_path = 'outputs/10.png'
    lidar_util.save_bev_image_by_point_cloud(pc_data, out_path)

    # # 2. 3d障碍物感知服务结果可视化
    # pcd_file = 'examples/object-aicv/1.pcd'
    # out_path = 'outputs/11.png'
    # infer(pcd_file, out_path, vis_mode=True)

    # 3. 3d目标检测结果可视化
    pcd_file = 'examples/object-aicv/1.pcd'
    label_file = 'examples/object-aicv/1.json'
    out_path = 'outputs/12.png'
    save_bev_img_with_bbox(pcd_file, label_file, out_path, vis_mode=True)

    # 4. 点云bin文件可视化
    bin_file = 'examples/semantic-kitti/velodyne/000000.bin'
    pc_data = lidar_util.load_bin_scan(bin_file)
    out_path = 'outputs/20.png'
    lidar_util.save_bev_image_by_point_cloud(pc_data, out_path)

    # 5. semantic-aicv可视化
    bin_file = 'examples/semantic-aicv/velodyne/000000.bin'
    label_file = 'examples/semantic-aicv/labels/000000.label'
    pc_data = lidar_util.load_bin_scan(bin_file)
    label = lidar_util.load_semantic_label(label_file)
    out_path = 'outputs/21.png'
    with open('configs/aicv.yaml', 'r') as stream:
        aicvaml = yaml.safe_load(stream)
    color_map = aicvaml['color_map']
    lidar_util.show_lidar_topview_with_semantic(pc_data, label, color_map, out_path, num_classes=8)

    # 6. semantic-kitti可视化
    bin_file = 'examples/semantic-kitti/velodyne/000000.bin'
    label_file = 'examples/semantic-kitti/labels/000000.label'
    pc_data = lidar_util.load_bin_scan(bin_file)
    label = lidar_util.load_semantic_label(label_file)
    out_path = 'outputs/22.png'
    with open('configs/semantic-kitti.yaml', 'r') as stream:
        aicvaml = yaml.safe_load(stream)
    color_map = aicvaml['color_map']
    lidar_util.show_lidar_topview_with_semantic(pc_data, label, color_map, out_path, num_classes=len(color_map)-1)

    # 7. 使用mayavi工具进行lidar数据可视化
    bin_file = 'examples/semantic-aicv/velodyne/000000.bin'
    label_file = 'examples/semantic-aicv/labels/000000.label'
    pc_data = lidar_util.load_bin_scan(bin_file)
    label = lidar_util.load_semantic_label(label_file)
    out_path = 'outputs/30.png'
    lidar_util.draw_lidar(pc_data, vis_output_file=out_path) # 纯lidar可视化，跟车视角
    out_path = 'outputs/31.png'
    lidar_util.draw_lidar(pc_data, label, vis_output_file=out_path) # 点云分割可视化，跟车视角
    out_path = 'outputs/32.png'
    lidar_util.draw_lidar(pc_data, label, vis_output_file=out_path, show_top_view=True) # 点云分割可视化，bev视角

    # 8. 使用mayavi工具进行lidar数据可视化，加bbox3d框
    pcd_file = 'examples/object-aicv/1.pcd'
    label_file = 'examples/object-aicv/1.json'
    out_path = 'outputs/13.png'
    save_bev_img_with_bbox(pcd_file, label_file, out_path, use_mayavi=True, vis_mode=True)
