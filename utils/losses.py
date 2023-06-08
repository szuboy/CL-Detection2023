# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


def focal_loss(predict, target):

    # clip predict heatmap range to prevent loss from becoming nan | 由于log2操作，限制热图预测值范围，防止loss NAN
    predict = torch.clamp(predict, min=1e-4, max=1-1e-4)

    pos_inds = target.gt(0.9)
    neg_inds = target.lt(0.9)
    neg_weights = torch.pow(1 - target[neg_inds], 4)

    pos_pred = predict[pos_inds]
    neg_pred = predict[neg_inds]

    pos_loss = torch.log2(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log2(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def load_loss(loss_name):
    if loss_name == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'focalLoss':
        loss_fn = focal_loss
    else:
        raise ValueError('Please input valid model name, {} not in loss zone.'.format(loss_name))
    return loss_fn


if __name__ == '__main__':
    loss_fn = load_loss(loss_name='L1')
    print(loss_fn)
