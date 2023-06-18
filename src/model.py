import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import math
from torchvision.ops.focal_loss import sigmoid_focal_loss as FocalLoss

class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size()).cuda()
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss
    
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine
    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


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
    

class customNet(nn.Module):

    def __init__(self, model_cfg):
        super(Net, self).__init__()

        self.backbone = timm.create_model('convnext_tiny',
                                        pretrained=True, 
                                        num_classes=0, 
                                        drop_rate=0.7,
                                        drop_path_rate=0.4, 
                                        in_chans=3)
    
        backbone_out = 768
        self.SPoC = nn.AdaptiveAvgPool2d((1,1))
        self.GEM = GeM()
        self.MAC = nn.AdaptiveMaxPool2d((1,1))
        
        self.m_head = torch.nn.Linear(backbone_out, 512)
        self.g_head = torch.nn.Linear(backbone_out, 512)
        self.s_head = torch.nn.Linear(backbone_out, 512)
        self.head = ArcMarginProduct(768, 3)
        #self.head = torch.nn.Linear(768, 3)

    def forward(self, batch):
        x = self.backbone(batch['signals'])

        # m = self.MAC(x)
        # m = m[:,:,0,0]
        # g = self.GEM(x)
        # g = g[:,:,0,0]
        # s = self.SPoC(x)
        # s = s[:,:,0,0]

        # m = self.m_head(m)
        # g = self.g_head(g)
        # s = self.s_head(s)

        # fuse = torch.concat((m, g, s), dim=1)
        logit = self.head(x)
        
        output = {}
        if self.training :
            
            #output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['label'], pos_weight=torch.tensor([12, 12, 12]).cuda())
            output['focal_loss'] = FocalLoss(logit, batch['label'], gamma=2)
            #output['arc_loss'] = ArcFaceLoss()(logits=logit, labels=batch['class'].unsqueeze(1).long())
        else :
            #output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['label'], pos_weight=torch.tensor([12, 12, 12]).cuda())
            output['focal_loss'] = FocalLoss(logit, batch['label'], gamma=2)
            #output['arc_loss'] = ArcFaceLoss()(logits=logit, labels=batch['class'].unsqueeze(1).long())
            output['probability'] = torch.sigmoid(logit)
            output['label'] = batch['label']
        return output
