'''
File Created: Thursday, 11th April 2019 6:44:00 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''

import torch
import torch.nn as nn

class FCN3DLoss(nn.Module):
    """

    """
    def __init__(self, alpha=1, beta=1.5, eta=1, gamma=2):
        super(FCN3DLoss, self).__init__()
        # self.iden_mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        if gamma == 0:
            print("set_loss: Normal Loss")
        elif gamma > 0:
            print("set_loss: Focal Loss, Gamma is {}".format(gamma))
        else:
            raise NotImplementedError        

    def forward(self, est, gt):
        # iden_loss = self.iden_mse(est*masks, gt)
        g_map_sum = torch.sum(gt['obj']) # N, C, D, H, W
        non_g_map_sum = 1
        for itm in gt['obj'].size():
            non_g_map_sum *= itm
        non_g_map_sum = non_g_map_sum - g_map_sum
        non_g_map = torch.ones_like(gt['obj']) - gt['obj']
        elosion = 1e-6
        is_obj_loss = -(torch.sum(
            gt['obj'] * (1 - est['obj'] + elosion) ** self.gamma * torch.log(est['obj'] + elosion))
                        / (g_map_sum+elosion) * self.alpha)
        non_obj_loss = -(torch.sum(
            non_g_map * (est['obj'] + elosion) ** self.gamma * torch.log(1 - est['obj'] + elosion))
                         / (non_g_map_sum+elosion) * self.beta)
        obj_loss = (is_obj_loss + non_obj_loss) * self.eta

        reg_loss = gt['obj'] * torch.sum((gt['reg'] - est['reg']) ** 2, dim=1)
        reg_loss = torch.sum(reg_loss * 0.02)

        loss = obj_loss + reg_loss
        return loss