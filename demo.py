# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:11:07 2018

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
from datasets.datasets import ImageFolder
from utils.utils import non_max_suppression
from config import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
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
  
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

dataloader = DataLoader(ImageFolder(cfg['test_root'], cfg['test_path'], img_size=cfg['img_shape'],transform=transform),
                        batch_size=opt.batch_size, shuffle=True)


classes = ['b','s']  # Extracts class labels from file

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection: %d samples...'%len(dataloader))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = input_imgs.type(Tensor)

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)  ### 8 + 2 + 1 + cls
        # 8 + 2 + cls + 4_bits
        detections = non_max_suppression(detections, cfg["num_classes"], conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)  ##0-7: 8bit coors, 8: conf, 9:cls_conf, 10: class, 11-14: 4bit coors.
    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)
    if batch_i == 100:
        break
    
# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)][1:]


bbox_colors = random.sample(colors, cfg["num_classes"])
print ('\nSaving images:')

for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
#
    print ("(%d) Image: '%s'" % (img_i, path))
    # Create plot
    img = np.array(Image.open(path))
    img_name = path.split(" ")[0].split("/")[-1]
    
    pad_x = max(img.shape[0] - img.shape[1],0) * (cfg['img_shape'] / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0],0) * (cfg['img_shape'] / max(img.shape))
    
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
            color_ = [[0,255,0],[255,0,0]]
            img = cv2.polylines(img, [coor], True, color_[int(label)], thickness=2)
           # cls_conf = memo[8].item()
            
            #cv2.putText(img, classes[int(label)], tuple(coor[0][0]), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color)
            
        cv2.imwrite(r'output/%s'%(img_name), img)
            
        

       
        
  
