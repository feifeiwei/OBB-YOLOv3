# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .layers import conv_bn,DarknetBlock,detect_layer, loss_layer
import pdb

# kaiming_weights_init
def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0
            
class Darknet53(nn.Module):
    def __init__(self, num_blocks):
        super(Darknet53,self).__init__()
        self.conv = conv_bn(3, 32, kernel=3, stride=1, padding=1)
        
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[4], stride=2)

    def _make_layer(self, ch_in, num_blocks, stride=1):
        layers = [conv_bn(ch_in, ch_in*2, stride=stride, padding=1)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers) 

    def forward(self, x):
        #pdb.set_trace()
        out = self.conv(x)
        
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return (c3, c4, c5)
    

class yolo(nn.Module):
    def __init__(self,num_blocks, anchors, input_dim, num_classes,use_cuda=False):
        super(yolo,self).__init__()
        self.extractor = Darknet53(num_blocks)
        #pdb.set_trace()
        self.predict_conv_list1 = nn.ModuleList(predict_conv_list1(num_classes))
        self.predict_conv_list2 = nn.ModuleList(predict_conv_list2(num_classes))
        self.predict_conv_list3 = nn.ModuleList(predict_conv_list3(num_classes))
        self.smooth_conv1 = conv_bn(512, 256, kernel=1, stride=1, padding=0)
        self.smooth_conv2 = conv_bn(256, 128, kernel=1, stride=1, padding=0)
        
        self.detection = detect_layer(anchors,input_dim,num_classes,use_cuda)
        self.loss0 = loss_layer(anchors[0], input_dim, num_classes)
        self.loss1 = loss_layer(anchors[1], input_dim, num_classes)
        self.loss2 = loss_layer(anchors[2], input_dim, num_classes)
        
    def _make_layer(self,in_planes,num_block,stride=1):
        layers = [conv_bn(in_planes,2*in_planes,kernel=3,stride=stride,padding=1)]
        for i in range(num_block):
            layers.append(DarknetBlock(2*in_planes))
        return nn.Sequential(*layers)
    def forward(self,x, target=None):
        c3, c4, c5 = self.extractor(x)
        #pdb.set_trace()
        
        x = c5                 
        for i in range(5):
            x = self.predict_conv_list1[i](x)  
        sm1 = self.smooth_conv1(x)  
        sm1 = F.upsample(sm1,scale_factor=2, mode='nearest') 
        sm1 = torch.cat((sm1,c4),1) 
        for i in range(5,7):
            x = self.predict_conv_list1[i](x)
        out1 = x                                   
        
        x = sm1    
        for i in range(5):
            x = self.predict_conv_list2[i](x)  
        sm2 = self.smooth_conv2(x)  
        sm2 = F.upsample(sm2,scale_factor=2)  
        sm2 = torch.cat((sm2,c3),1)           
        for i in range(5,7):
            x = self.predict_conv_list2[i](x)
        out2 = x                                      
        
        x = sm2   
        for i in range(7):
            x = self.predict_conv_list3[i](x)
        out3 = x                                                 
        
        if target is None:
            detections = self.detection(out1,out2,out3)
            return detections
        else:
            loss_0 = self.loss0(out1,target)
            loss_1 = self.loss1(out2,target)
            loss_2 = self.loss2(out3,target)
            
            ret = []
            for i in zip(loss_0,loss_1,loss_2):
                ret.append(sum(i))
            return ret
    
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.extractor.load_state_dict(torch.load(base_file))
            print("initing  darknet53 ......")
            self.predict_conv_list1.apply(weights_init)
            self.smooth_conv1.apply(weights_init)
            self.predict_conv_list2.apply(weights_init)
            self.smooth_conv2.apply(weights_init)
            self.predict_conv_list3.apply(weights_init)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
    
def predict_conv_list1(num_classes):
    layers = []
    layers += [conv_bn(1024, 512, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(512, 1024, kernel=3, stride=1, padding=1)]
    layers += [conv_bn(1024, 512, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(512, 1024, kernel=3, stride=1, padding=1)]
    layers += [conv_bn(1024, 512, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(512, 1024, kernel=3, stride=1, padding=1)]
    layers += [nn.Conv2d(1024, (11 + num_classes) * 3, kernel_size=1, stride=1, padding=0)]
    return layers

def predict_conv_list2(num_classes):
    layers = list()
    layers += [conv_bn(768, 256, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(256, 512, kernel=3, stride=1, padding=1)]
    layers += [conv_bn(512, 256, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(256, 512, kernel=3, stride=1, padding=1)]
    layers += [conv_bn(512, 256, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(256, 512, kernel=3, stride=1, padding=1)]
    layers += [nn.Conv2d(512, (11 + num_classes) * 3, kernel_size=1, stride=1, padding=0)]
    return layers

def predict_conv_list3(num_classes):
    layers = list()
    layers += [conv_bn(384, 128, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(128, 256, kernel=3, stride=1, padding=1)]
    layers += [conv_bn(256, 128, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(128, 256, kernel=3, stride=1, padding=1)]
    layers += [conv_bn(256, 128, kernel=1, stride=1, padding=0)]
    layers += [conv_bn(128, 256, kernel=3, stride=1, padding=1)]
    layers += [nn.Conv2d(256, (11 + num_classes) * 3, kernel_size=1, stride=1, padding=0)]
    return layers

def yolov3(input_dim, anchors, num_classes,cuda=True):
    num_blocks = [1,2,8,8,4]
    return yolo(num_blocks, anchors, input_dim, num_classes, cuda)

if __name__=="__main__":
    
    anchors = [[(116, 90), (156, 198), (373, 326)], 
                [(30, 61), (62, 45), (59, 119)], 
                [(10, 13), (16, 30), (33, 23)]]
    
    shape = 416
    x = torch.randn(1,3,shape,shape)
    n = yolov3(shape, anchors, 20, cuda=False)
    y = n(x) 
