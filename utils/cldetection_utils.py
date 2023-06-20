# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import shutil
import numpy as np
import SimpleITK as sitk
from skimage import io as sk_io
from skimage import draw as sk_draw


def check_and_make_dir(dir_path: str) -> None:
    """
    function to create a new folder, if the folder path dir_path in does not exist
    :param dir_path: folder path | 文件夹路径
    :return: None
    """
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError('Error, the provided path (%s) is a file path, not a folder path.' % dir_path)
        shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


def load_train_stack_data(file_path: str) -> np.ndarray:
    """
    function to load train_stack.mha data file | 加载train_stack.mha数据文件的函数
    :param file_path: train_stack.mha filepath | 挑战赛提供的train_stack.mha文件路径
    :return: a 4-dim array containing 400 training set cephalometric images | 一个包含了400张训练集头影图像的四维的矩阵
    """
    sitk_stack_image = sitk.ReadImage(file_path)
    np_stack_array = sitk.GetArrayFromImage(sitk_stack_image)
    return np_stack_array


def remove_zero_padding(image_array: np.ndarray) -> np.ndarray:
    """
    function to remove zero padding in an image | 去除图像中的0填充函数
    :param image_array: one cephalometric image array, shape is (2400, 2880, 3) | 一张头影图像的矩阵，形状为(2400, 2880, 3)
    :return: image matrix after removing zero padding | 去除零填充部分的图像矩阵
    """
    row = np.sum(image_array, axis=(1, 2))
    column = np.sum(image_array, axis=(0, 2))

    non_zero_row_indices = np.argwhere(row != 0)
    non_zero_column_indices = np.argwhere(column != 0)

    last_row = int(non_zero_row_indices[-1])
    last_column = int(non_zero_column_indices[-1])

    image_array = image_array[:last_row+1, :last_column+1, :]
    return image_array


def calculate_prediction_metrics(result_dict: dict):
    """
    function to calculate prediction metrics | 计算评价指标
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
    :return: MRE and 2mm SDR metrics
    """
    n_landmarks = 0
    sdr_landmarks = 0
    n_landmarks_error = 0
    for file_path, landmark_dict in result_dict.items():
        scale = landmark_dict['scale']
        landmarks, predict_landmarks = landmark_dict['gt'], landmark_dict['predict']

        # landmarks number
        n_landmarks = n_landmarks + np.shape(landmarks)[0]

        # mean radius error (MRE)
        each_landmark_error = np.sqrt(np.sum(np.square(landmarks - predict_landmarks), axis=1)) * scale
        n_landmarks_error = n_landmarks_error + np.sum(each_landmark_error)

        # 2mm success detection rate (SDR)
        sdr_landmarks = sdr_landmarks + np.sum(each_landmark_error < 2)

    mean_radius_error = n_landmarks_error / n_landmarks
    sdr = sdr_landmarks / n_landmarks

    print('Mean Radius Error (MRE): {}, 2mm Success Detection Rate (SDR): {}'.format(mean_radius_error, sdr))
    return mean_radius_error, sdr


def visualize_prediction_landmarks(result_dict: dict, save_image_dir: str):
    """
    function to visualize prediction landmarks  | 可视化预测结果
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
    :param save_image_dir: the folder path to save images
    :return: None
    """
    for file_path, landmark_dict in result_dict.items():
        landmarks, predict_landmarks = landmark_dict['gt'], landmark_dict['predict']

        image = sk_io.imread(file_path)
        image_shape = np.shape(image)[:2]

        for i in range(np.shape(landmarks)[0]):
            landmark, predict_landmark = landmarks[i, :], predict_landmarks[i, :]
            # ground truth landmark
            radius = 7
            rr, cc = sk_draw.disk(center=(int(landmark[1]), int(landmark[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [255, 0, 0]
            # model prediction landmark
            rr, cc = sk_draw.disk(center=(int(predict_landmark[1]), int(predict_landmark[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [0, 255, 0]
            # the line between gt landmark and prediction landmark
            line_width = 5
            rr, cc, value = sk_draw.line_aa(int(landmark[1]), int(landmark[0]), int(predict_landmark[1]), int(predict_landmark[0]))
            for offset in range(line_width):
                offset_rr, offset_cc = np.clip(rr + offset, 0, image_shape[0] - 1), np.clip(cc + offset, 0, image_shape[1] - 1)
                image[offset_rr, offset_cc, :] = [255, 255, 0]

        filename = os.path.basename(file_path)
        sk_io.imsave(os.path.join(save_image_dir, filename), image)


