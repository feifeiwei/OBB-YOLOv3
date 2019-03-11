# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:47:03 2018

@author: 60236
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import predict_transform, get_target

class conv_bn(nn.Module):
    def __init__(self,in_planes, planes, kernel=3, stride=1, padding=1,bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes,planes,kernel_size=kernel,stride=stride,padding=padding,bias=bias)
        self.bn = nn.BatchNorm2d(planes, momentum=0.01, eps=1e-05, affine=True)
    def forward(self,x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1, inplace=True)
    
    
class DarknetBlock(nn.Module):  #   1*1  减少通道数   3*3增加通道
    def __init__(self,in_planes):
        super().__init__()
        mid_ch = in_planes // 2
        self.conv1 = conv_bn(in_planes,mid_ch, kernel=1,stride=1,padding=0)
        self.conv2 = conv_bn(mid_ch,in_planes,kernel=3,stride=1,padding=1)
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x
    
class detect_layer(nn.Module):
    def __init__(self,anchors, input_dim, num_classes,use_cuda=True):
        super(detect_layer,self).__init__()
        self.anchors = anchors
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_cuda = use_cuda
    def forward(self, x,y,z):
        prediction_x = predict_transform(x, self.input_dim, self.anchors[0], self.num_classes, self.use_cuda)
        prediction_y = predict_transform(y, self.input_dim, self.anchors[1], self.num_classes, self.use_cuda)
        prediction_z = predict_transform(z, self.input_dim, self.anchors[2], self.num_classes, self.use_cuda)
        prediction = torch.cat((prediction_x, prediction_y, prediction_z), 1) #1,10647,85
        return prediction
    
class loss_layer(nn.Module):
    def __init__(self, anchors, img_size, num_classes):
        super(loss_layer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 11 + num_classes
        
        self.ignore_threshold = 0.5
        self.lambda_xy = 1.
        self.lambda_coors = 1.
        self.lambda_conf = 5.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.smmoth_l1_loss = nn.SmoothL1Loss()
        self.img_size = img_size
        
    def forward(self, fms, targets):
        '''
            fms:  ?, 255,13,13   / (26,26)/(52,52)
            target:  ?,100,11
        '''
        bs = fms.size(0)
        fm_size = fms.size(2)
        stride = self.img_size / fm_size
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        # ?,3,13,13,25
        prediction = fms.view(bs,  self.num_anchors, 
                                self.bbox_attrs, fm_size, fm_size).permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        x = torch.sigmoid(prediction[..., 8])          # Center x
        y = torch.sigmoid(prediction[..., 9])          # Center y
        x1 = prediction[..., 0]
        y1 = prediction[..., 1]
        x2 = prediction[..., 2]
        y2 = prediction[..., 3]
        x3 = prediction[..., 4]
        y3 = prediction[..., 5]
        x4 = prediction[..., 6]
        y4 = prediction[..., 7]
        print('x:',x.shape)
        

        conf = torch.sigmoid(prediction[..., 10])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 11:]) # Cls pred.
        
      
            #  build target
        mask, conf_mask, tx, ty, tx1, ty1, tx2, ty2,tx3, ty3,tx4, ty4, tconf, tcls = get_target(targets, scaled_anchors,
                                                                                               fm_size,
                                                                                               self.ignore_threshold,
                                                                                               self.num_classes)
        mask, conf_mask = mask.byte().cuda(), conf_mask.byte().cuda()
        tx, ty = tx.cuda(), ty.cuda()
        tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4 = tx1.cuda(), ty1.cuda(), tx2.cuda(), ty2.cuda(), tx3.cuda(), ty3.cuda(), tx4.cuda(), ty4.cuda()
        tconf, tcls = tconf.cuda(), tcls.cuda()
        print('tx:',tx.shape)
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask
                                  
        #  losses.
        loss_x = self.bce_loss(x[mask==1], tx[mask==1]) * self.lambda_xy
        loss_y = self.bce_loss(y[mask==1], ty[mask==1]) * self.lambda_xy
        
        loss_x1 = self.smmoth_l1_loss(x1[mask==1], tx1[mask==1]) * self.lambda_coors
        loss_y1 = self.smmoth_l1_loss(y1[mask==1], ty1[mask==1]) * self.lambda_coors 
        loss_x2 = self.smmoth_l1_loss(x2[mask==1], tx2[mask==1]) * self.lambda_coors
        loss_y2 = self.smmoth_l1_loss(y2[mask==1], ty2[mask==1]) * self.lambda_coors 
        loss_x3 = self.smmoth_l1_loss(x3[mask==1], tx3[mask==1]) * self.lambda_coors
        loss_y3 = self.smmoth_l1_loss(y3[mask==1], ty3[mask==1]) * self.lambda_coors 
        loss_x4 = self.smmoth_l1_loss(x4[mask==1], tx4[mask==1]) * self.lambda_coors
        loss_y4 = self.smmoth_l1_loss(y4[mask==1], ty4[mask==1]) * self.lambda_coors 
        
        loss_conf = self.bce_loss(conf[conf_mask_true], tconf[conf_mask_true]) +\
                    self.bce_loss(conf[conf_mask_false], tconf[conf_mask_false])
        loss_cls = self.bce_loss(pred_cls[mask==1],tcls[mask==1]) * self.lambda_cls
        
        loss = loss_x  + loss_y  + loss_x1  + loss_y1 + loss_x2 + loss_y2 + loss_x3 + loss_y3 + loss_x4 + loss_y4 + loss_conf  + loss_cls 

        return tuple((loss, loss_x.item(), loss_y.item(), loss_x1.item(),\
                loss_y1.item(), loss_x2.item(),loss_y2.item(),loss_x3.item(),loss_y3.item(),loss_x4.item(), loss_y4.item(), loss_conf.item(), loss_cls.item() ))
        
        


