'''
File Created: Wednesday, 1st May 2019 3:46:18 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import torch
import torch.nn as nn

class VoxelNetLoss(nn.Module):
    def __init__(self, alpha=1, beta=1.5, eta=1, gamma=2):
        '''
        Loss function for VoxelNet
        inputs:
            alpha: weight of pos cls
            bet: weight of neg cls
            eta: weight of cls
            gamma: hyperparameter of focal loss
        '''
        super(VoxelNetLoss, self).__init__()
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
        small_addon_for_BCE = 1e-6
        pos_equal_one_for_reg = torch.cat([gt["obj"][:, 0:1, :, :].repeat(1, 7, 1, 1), gt["obj"][:, 1:2, :, :].repeat(1, 7, 1, 1)], dim=1)
        pos_equal_one_sum = torch.clamp(torch.sum(gt['obj']), min=1)
        neg_equal_one_sum = torch.clamp(torch.sum(gt['neg-obj']), min=1)
        cls_pos_loss = (-gt["obj"] * (1 - est["obj"] + small_addon_for_BCE) ** self.gamma * torch.log(    est["obj"] + small_addon_for_BCE)) / pos_equal_one_sum * self.alpha
        cls_neg_loss = (-gt["neg-obj"] * (est["obj"] + small_addon_for_BCE) ** self.gamma * torch.log(1 - est["obj"] + small_addon_for_BCE)) / neg_equal_one_sum * self.beta
        cls_loss = torch.sum(cls_pos_loss + cls_neg_loss) * self.eta
        cls_pos_loss = torch.sum(cls_pos_loss)
        cls_neg_loss = torch.sum(cls_neg_loss)
        reg_loss = smooth_l1(est["reg"] * pos_equal_one_for_reg, gt["reg"] * pos_equal_one_for_reg, sigma=3) / pos_equal_one_sum
        reg_loss = torch.sum(reg_loss)
        loss = torch.sum(cls_loss + reg_loss)
        losses = dict()
        losses["loss"] = loss
        losses["cls_loss"] = cls_loss
        losses["reg_loss"] = reg_loss
        losses["cls_pos_loss"] = cls_pos_loss
        losses["cls_neg_loss"] = cls_neg_loss
        return losses

def smooth_l1(est, gt, sigma=3.0):
    '''
        smooth_l1 distance
    '''
    sigma2 = sigma * sigma
    diffs = est - gt
    smooth_l1_signs = torch.le(torch.abs(diffs), 1.0 / sigma2).float()

    smooth_l1_option1 = diffs * diffs * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = smooth_l1_option1 * smooth_l1_signs + smooth_l1_option2 * (1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add
    return smooth_l1