import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import math
from torchvision.ops.focal_loss import sigmoid_focal_loss as FocalLoss

class Net(nn.Module):

    def __init__(self,
                model_cfg,
                ):
        super(Net, self).__init__()
        model_name = model_cfg.get('model_name', 'tf_efficientnet_b4')
        pretrained = model_cfg.get('pretrained',False)
        in_chans = model_cfg.get('in_chans' , 3)
        num_classes = model_cfg.get('num_classes', 1)
        drop_rate = model_cfg.get('drop_rate', 0.2)
        drop_path_rate = model_cfg.get('drop_path_rate', 0.2)
        self.model = timm.create_model(model_name = model_name, 
                                       pretrained=pretrained, 
                                       in_chans = in_chans, 
                                       num_classes = num_classes,
                                       drop_rate = drop_rate,
                                       drop_path_rate=drop_path_rate)
    
    def forward(self, batch):
        x = batch['signals']
        B, C, H, W = x.shape
        logit = self.model(x)

        output = {}
        if self.training :
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['label'], pos_weight=torch.tensor([12, 12, 12]).cuda())
            output['focal_loss'] = FocalLoss(logit, batch['label'], gamma=2)
        else :
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['label'], pos_weight=torch.tensor([12, 12, 12]).cuda())
            output['focal_loss'] = FocalLoss(logit, batch['label'], gamma=2)
            output['probability'] = torch.sigmoid(logit)
            output['label'] = batch['label']
        return output