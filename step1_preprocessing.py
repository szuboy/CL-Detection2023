# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhanghongyuan2017@email.szu.edu.cn

import os
import json
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk

from utils.cldetection_utils import load_train_stack_data, remove_zero_padding, check_and_make_dir


def extract_one_image_landmarks(all_gt_dict: dict, image_id: int) -> dict:
    """
    function to extract landmark information corresponding to an image | 提出出对应图像的关键点信息
    :param all_gt_dict: a dict loaded from the train_gt.json file | 从train_gt.json文件加载得到的字典
    :param image_id: image id between 0 and 400 | 图像的id，在0到400之间
    :return: a dict containing pixel spacing and coordinates of 38 landmarks | 一个字典，包含了像素的spacing和38个关键点的坐标
    """
    image_dict = {'image_id': image_id}
    for landmark in all_gt_dict['points']:
        point = landmark['point']
        if point[-1] != image_id:
            continue
        image_dict['scale'] = float(landmark['scale'])
        image_dict['landmark_%s' % landmark['name']] = point[:2]
    return image_dict


def save_landmarks_list_as_csv(image_landmarks_list: list, save_csv_path: str, image_dir_path: str, image_suffix: str):
    """
    function to save the landmarks list corresponding to different images in a csv file | 将不同的图像的关键点以csv文件保存下来
    :param image_landmarks_list: a list of landmark annotations, each element is an annotation of an image | 关键点列表，每一个元素就是一个图片的标注
    :param save_csv_path: csv file save path | csv文件保存路径
    :return: None
    """
    # CSV header
    columns = ['file', 'scale']
    for i in range(38):
        columns.extend(['p{}x'.format(i + 1), 'p{}y'.format(i + 1)])
    df = pd.DataFrame(columns=columns)
    # CSV content
    for landmark in image_landmarks_list:
        row_line = [os.path.join(image_dir_path, str(landmark['image_id']) + image_suffix), landmark['scale']]
        for i in range(38):
            point = landmark['landmark_%s' % (i + 1)]
            row_line.extend([point[0], point[1]])
        df.loc[len(df.index)] = row_line
    # CSV writer
    df.to_csv(save_csv_path, index=False)


if __name__ == '__main__':
    # load the train_stack.mha data file using SimpleITK package
    # 使用SimpleITk库加载提供的train_stack.mha数据文件
    # TODO: Please remember to modify it to the data file path on your computer. | 请记得修改为自己电脑上的数据路径.
    mha_file_path = '/data/zhangHY/CL-Detection2023/train_stack.mha'
    train_stack_array = load_train_stack_data(mha_file_path)

    # The function of the following script is to remove the redundant 0 padding problem.
    # Don't worry, this operation will not affect the processing of the label points of the key points,
    # because the coordinates of the key points are all in the upper left corner as the reference system
    # 接下来的这段脚本的功能是去除多余的0填充问题
    # 放心，这个操作不会影响到关键点的标注点的处理，因为关键点的坐标都是左上角为参考系的
    # TODO: Please remember to modify it to the save dir path on your computer. | 请记得修改为自己电脑上的数据保存路径.
    save_dir_path = '/data/zhangHY/CL-Detection2023/processed_images'
    check_and_make_dir(save_dir_path)
    for i in range(np.shape(train_stack_array)[0]):
        image_array = train_stack_array[i, :, :, :]
        image_array = remove_zero_padding(image_array)
        pillow_image = Image.fromarray(image_array)
        pillow_image.save(os.path.join(save_dir_path, '%s.bmp' % (i + 1)))

    # load the train_gt.json annotation file using json package
    # 使用json库加载提供的train_gt.json标注文件
    # TODO: Please remember to modify it to the json file path on your computer. | 请记得修改为自己电脑上的标签JSON数据路径.
    json_file_path = '/data/zhangHY/CL-Detection2023/train-gt.json'
    with open(json_file_path, mode='r', encoding='utf-8') as f:
        train_gt_dict = json.load(f)

    # parse out the landmark annotations corresponding to each image
    # 解析出来每个图像对应的关键点标注
    all_image_landmarks_list = []
    for i in range(400):
        image_landmarks = extract_one_image_landmarks(all_gt_dict=train_gt_dict, image_id=i+1)
        all_image_landmarks_list.append(image_landmarks)

    # shuffle the order of the landmark annotations list
    # 打乱关键点列表的顺序
    random.seed(2023)
    random.shuffle(all_image_landmarks_list)

    # split the training set, validation set and test set, and save them as csv files
    # 划分训练集，验证集和测试集，并以csv文件形式保存
    train_csv_path = os.path.join(os.path.dirname(save_dir_path), 'train.csv')
    print('Train CSV Path:', train_csv_path)
    save_landmarks_list_as_csv(image_landmarks_list=all_image_landmarks_list[:300],
                               save_csv_path=train_csv_path,
                               image_dir_path=save_dir_path,
                               image_suffix='.bmp')

    valid_csv_path = os.path.join(os.path.dirname(save_dir_path), 'valid.csv')
    print('Valid CSV Path:', valid_csv_path)
    save_landmarks_list_as_csv(image_landmarks_list=all_image_landmarks_list[300:350],
                               save_csv_path=valid_csv_path,
                               image_dir_path=save_dir_path,
                               image_suffix='.bmp')

    test_csv_path = os.path.join(os.path.dirname(save_dir_path), 'test.csv')
    print('Test CSV Path:', test_csv_path)
    save_landmarks_list_as_csv(image_landmarks_list=all_image_landmarks_list[350:400],
                               save_csv_path=test_csv_path,
                               image_dir_path=save_dir_path,
                               image_suffix='.bmp')


