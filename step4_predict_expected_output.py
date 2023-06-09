# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tqdm
import json
import torch
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import io as sk_io
from skimage import transform as sk_transform

import warnings
warnings.filterwarnings('ignore')

from utils.model import load_model
from utils.cldetection_utils import load_train_stack_data, remove_zero_padding


def main(config):
    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # load model
    model = load_model(model_name=config.model_name)
    model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
    model = model.to(device)

    # load test.csv
    stacked_image_array = load_train_stack_data(config.load_mha_path)

    # test result dict
    all_images_predict_landmarks_list = []

    # test mode
    with torch.no_grad():
        model.eval()
        for i in range(np.shape(stacked_image_array)[0]):
            # one image array
            image = np.array(stacked_image_array[i, :, :, :])

            # remove zero padding
            image = remove_zero_padding(image)
            height, width = np.shape(image)[:2]

            # resize
            scaled_image = sk_transform.resize(image, (512, 512), mode='constant', preserve_range=False)

            # transpose channel and add batch-size channel
            transpose_image = np.transpose(scaled_image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)

            # model predict
            predict_heatmap = model(torch_image)

            # decode landmark location from heatmap
            predict_heatmap = predict_heatmap.detach().cpu().numpy()
            predict_heatmap = np.squeeze(predict_heatmap)

            landmarks_list = []
            for i in range(np.shape(predict_heatmap)[0]):
                # 索引得到不同的关键点热图
                landmark_heatmap = predict_heatmap[i, :, :]
                yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
                # there may be multiple maximum positions, and a simple average is performed as the final result
                x0, y0 = np.mean(xx), np.mean(yy)
                # zoom to original image size
                x0, y0 = x0 * width / 512, y0 * height / 512
                # append to landmarks list
                landmarks_list.append([x0, y0])
            all_images_predict_landmarks_list.append(landmarks_list)

    # save as expected format JSON file
    json_dict = {'name': 'Orthodontic landmarks', 'type': 'Multiple points'}

    all_predict_points_list = []
    for image_id, predict_landmarks in enumerate(all_images_predict_landmarks_list):
        for landmark_id, landmark in enumerate(predict_landmarks):
            points = {'name': str(landmark_id + 1),
                      'point': [landmark[0], landmark[1], image_id + 1]}
            all_predict_points_list.append(points)
    json_dict['points'] = all_predict_points_list

    # version information
    major = 1
    minor = 0
    json_dict['version'] = {'major': major, 'minor': minor}

    # JSON dict to JSON string
    json_string = json.dumps(json_dict, indent=4)
    with open(config.save_json_path, "w") as f:
        f.write(json_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters | 数据文件路径
    parser.add_argument('--load_mha_path', type=str, default='/home/medai06/zhangHY/CL-Detection2023/step5_docker_and_upload/test/stack1.mha')
    parser.add_argument('--save_json_path', type=str, default='/home/medai06/zhangHY/CL-Detection2023/step5_docker_and_upload/test/expected_output.json')

    # model load dir path | 存放模型的文件夹路径
    parser.add_argument('--load_weight_path', type=str, default='/data/zhangHY/CL-Detection2023/UNet_checkpoint/best_model.pt')

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='UNet')

    experiment_config = parser.parse_args()
    main(experiment_config)
