# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:14:18 2019

@author: 60236
"""
import cv2
import numpy as np
import os
import random
import time
import datetime
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib.ticker import NullLocator
from module.model import yolov3
from datasets.datasets import ListDataset
from utils.utils import non_max_suppression
from config import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.1, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=2, help='size of the batches')

opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available else False
os.makedirs('output', exist_ok=True)



cfg = Dota_config

model = yolov3(cfg['img_shape'], cfg['anchors'], cfg['num_classes']).to(device)
checkpoint = torch.load(r'./checkpoints/%s'%(cfg['save_name']))
model.load_state_dict(checkpoint['weights'])
print('loading model weights success')
  
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

dataloader = DataLoader(ListDataset(cfg['test_path'], img_size=cfg['img_shape'],transform=transform,train=False),
                        batch_size=opt.batch_size, shuffle=False)


classes = ['b','s']  # Extracts class labels from file

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection: %d samples...'%len(dataloader))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()


print ('Compute mAP...')

all_detections = []
all_annotations = []

for batch_i, (_, imgs, targets) in enumerate(dataloader):
    # Configure input
    imgs = imgs.type(Tensor)

    # Get detections
    with torch.no_grad():
        outputs = model(imgs)  ### 8 + 2 + 1 + cls
        ##0-7: 8bit coors, 8: conf, 9:cls_conf, 10: class, 11-14: 4bit coors.
        outputs = non_max_suppression(outputs, cfg["num_classes"], conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)  
        tmp = []
        for o in outputs:
            coor_4bit = o[:,-4:]
            conf_cls = o[:,8:11]
            tmp.append(torch.cat((coor_4bit,conf_cls),1))
            print(torch.cat((coor_4bit,conf_cls),1).shape)
        outputs = tmp
        
    # Save image and detections
    #imgs.extend(img_paths)
    #img_detections.extend(detections)
    break
