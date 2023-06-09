# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhanghongyuan2017@email.szu.edu.cn

import os
import tqdm
import torch
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

from utils.tranforms import Rescale, RandomHorizontalFlip, ToTensor
from utils.dataset import CephXrayDataset
from utils.model import load_model
from utils.losses import load_loss


from utils.cldetection_utils import check_and_make_dir


def main(config):
    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # train and valid dataset
    train_transform = transforms.Compose([Rescale(output_size=(config.image_height, config.image_width)),
                                          RandomHorizontalFlip(p=config.flip_augmentation_prob),
                                          ToTensor()])
    train_dataset = CephXrayDataset(csv_file_path=config.train_csv_path, transform=train_transform)
    valid_transform = transforms.Compose([Rescale(output_size=(config.image_height, config.image_width)),
                                          ToTensor()])
    valid_dataset = CephXrayDataset(csv_file_path=config.valid_csv_path, transform=valid_transform)

    # train and valid dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size_valid,
                              shuffle=False,
                              num_workers=config.num_workers)

    # load model
    model = load_model(model_name=config.model_name)
    model = model.to(device)

    # optimizer and StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(config.beta1, config.beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.scheduler_step_size,
                                                gamma=config.scheduler_gamma)

    # model loss function
    loss_fn = load_loss(loss_name=config.loss_name)

    # model training preparation
    train_losses = []
    valid_losses = []
    best_loss = 1e10
    num_epoch_no_improvement = 0
    check_and_make_dir(config.save_model_dir)

    # start to train and valid
    for epoch in range(config.train_max_epoch):
        scheduler.step(epoch)
        model.train()
        for (image, heatmap) in tqdm.tqdm(train_loader):
            image, heatmap = image.float().to(device), heatmap.float().to(device)
            output = model(image)
            loss = loss_fn(output, heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 3))
        print('Train epoch [{:<4d}/{:<4d}], Loss: {:.6f}'.format(epoch + 1, config.train_max_epoch, np.mean(train_losses)))

        # save model checkpoint
        if epoch % config.save_model_step == 0:
            torch.save(model.state_dict(), os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))
            print("Saving checkpoint model ", os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))

        # valid model, save best_checkpoint.pkl
        with torch.no_grad():
            model.eval()
            print("Validating....")
            for (image, heatmap) in tqdm.tqdm(valid_loader):
                image, heatmap = image.float().to(device), heatmap.float().to(device)
                output = model(image)
                loss = loss_fn(output, heatmap)
                valid_losses.append(loss.item())
        valid_loss = np.mean(valid_losses)
        print('Validation loss: {:.6f}'.format(valid_loss))

        # early stop mechanism
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.6f} to {:.6f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.save_model_dir, "best_model.pt"))
            print("Saving best model ", os.path.join(config.save_model_dir, "best_model.pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == config.epoch_patience:
            print("Early Stopping!")
            break

        # reset parameters
        train_losses = []
        valid_losses = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters | 数据文件路径
    parser.add_argument('--train_csv_path', type=str)
    parser.add_argument('--valid_csv_path', type=str)

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model training hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='UNet')
    parser.add_argument('--train_max_epoch', type=int, default=400)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_valid', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--save_model_step', type=int, default=2)

    # data augmentation
    parser.add_argument('--flip_augmentation_prob', type=float, default=0.5)

    # model loss function
    parser.add_argument('--loss_name', type=str, default='focalLoss')

    # early stop mechanism
    parser.add_argument('--epoch_patience', type=int, default=5)

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-4)

    # Adam optimizer parameters
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Step scheduler parameters
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)

    # result & save
    parser.add_argument('--save_model_dir', type=str, default='./model/')

    experiment_config = parser.parse_args()
    main(experiment_config)

