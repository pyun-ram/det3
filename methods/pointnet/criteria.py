'''
File Created: Monday, 12th August 2019 2:37:15 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetClsLoss(nn.Module):
    def __init__(self):
        super(PointNetClsLoss, self).__init__()
    def forward(self, est, gt):
        '''
        inputs:
            est: [#batch, lenth of one hot vector]
            gt: [#batch, ]
        '''
        loss = F.nll_loss(est, gt)
        print(est.shape, gt.shape)
        return loss

