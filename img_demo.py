# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:12:14 2019

@author: 60236
"""
import cv2
import glob
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
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from module.model import yolov3
from datasets.datasets import ImageFolder2
from utils.utils import non_max_suppression
from config import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.1, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')

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

if cuda:
    model.cuda()    
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

img_root = r'C:\Users\60236\Desktop\dota\1'
dataloader = DataLoader(ImageFolder2(img_root, img_size=cfg['img_shape'],transform=transform),
                        batch_size=1, shuffle=False)


classes = ['small','big']  # Extracts class labels from file

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection: %d samples...'% len(dataloader))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()

for batch_i, (image_path, input_img) in enumerate(dataloader):
    img = input_img.float().cuda()
    with torch.no_grad():
        detections = model(img)  ### 8 + 2 + 1 + cls
        detections = non_max_suppression(detections, cfg["num_classes"], conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
    
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(image_path)
    img_detections.extend(detections)
    
# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

bbox_colors = random.sample(colors, cfg["num_classes"])
print ('\nSaving images:')


for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
#
    print ("(%d) Image: '%s'" % (img_i, path))
    # Create plot
    img = np.array(Image.open(path))
    img_name = path.split(" ")[0].split("\\")[-1]
    
    pad_x = max(img.shape[0] - img.shape[1],0) * (1920 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0],0) * (1920 / max(img.shape))
    
    unpad_h = cfg['img_shape'] - pad_y
    unpad_w = cfg['img_shape'] - pad_x
    
    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, 10].cpu().unique()
        n_cls_preds = len(unique_labels)
        for memo in detections:
            coor = memo[:8]
            for i in range(8):
                if i%2 != 0:
                    coor[i] = ((coor[i] - pad_y //2) / unpad_h) * img.shape[0]
                else:
                    coor[i] = ((coor[i] - pad_x //2) / unpad_w) * img.shape[1]
            
            coor = coor.reshape((-1,1,2)).cpu().numpy().astype(np.int32)
            label = memo[10].item()
            color = bbox_colors[int(np.where(unique_labels == int(label))[0])]
            color = tuple(i*255 for i in color)
            img = cv2.polylines(img, [coor], True, (0,255,0), thickness=2)
            cls_conf = memo[8].item()
            
            cv2.putText(img, classes[int(label)], tuple(coor[0][0]), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color)
            
        cv2.imwrite('output/%s'%(img_name), img)
    
    
    
    
    
    









