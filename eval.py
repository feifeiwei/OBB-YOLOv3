# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:14:18 2019
@author: 60236
"""
import os
import tqdm
import time
import torch
import argparse
import numpy as np

from module.model import yolov3
from torch.utils.data import DataLoader
from datasets.datasets import ListDataset
from utils.utils import non_max_suppression,change_box_order,bbox_iou_numpy,compute_ap
from config import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.1, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')

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

for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    print(batch_i)
    # Configure input
    imgs = imgs.type(Tensor)
    targets = targets.type(Tensor)
#    # Get detections
    with torch.no_grad():
        outputs = model(imgs)  ### 8 + 2 + 1 + cls
        ##0-7: 8bit coors, 8: conf, 9:cls_conf, 10: class, 11-14: 4bit coors.
        outputs = non_max_suppression(outputs, cfg["num_classes"], conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)  
    tmp = []
    for o in outputs:
        try:
            coor_4bit = o[:,-4:]
            conf_cls = o[:,8:11]
        except:
            coor_4bit = torch.zeros((1,4))
            conf_cls = torch.zeros((1,3))
        tmp.append(torch.cat((coor_4bit,conf_cls),1))
    outputs = tmp
     ###8bit to 4bit for target
    targets[:,:,:8] *= cfg['img_shape'] 
    coor_8bit = targets[:,:,:8]
    coor_4bit = torch.zeros((coor_8bit.shape[0],coor_8bit.shape[1],4))
    for b, bit_8 in enumerate(coor_8bit):
        coor_4bit[b] = change_box_order(bit_8,'xiyi2xyxy')
#        print(change_box_order(bit_8,'xiyi2xyxy'))
#        print(bit_8)
     
    tmp = torch.cat((coor_4bit, targets[:,:,10:].cpu()),2)
    targets = tmp #[xm,ym,xmax,ymax, conf,cls_conf, class]

    for output, annotations in zip(outputs, targets): ## [?, 7]-----[50,5]////[b,?,5]
        all_detections.append([np.array([]) for _ in range(cfg['num_classes'])])
        if output is not None:
            pred_boxes = output[:, :5].cpu().numpy()
            scores = output[:, 4].cpu().numpy()
            pred_labels = output[:, -1].cpu().numpy()
#        
            # Order by confidence
            sort_i = np.argsort(scores)  #
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]
#            
            for label in range(cfg['num_classes']):
                all_detections[-1][label] = pred_boxes[pred_labels == label]
        all_annotations.append([np.array([]) for _ in range(cfg['num_classes'])])
        if any(annotations[:, 0] > 0):
            annotation_labels = annotations[annotations[:, 0] > 0, -1].numpy()
            _annotation_boxes = annotations[annotations[:, 0] > 0, :-1] # [xm.ym,xmax,ymax,conf,cls_conf]
#            
#            # Reformat to x1, y1, x2, y2 and rescale to image dimensions
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0]
            annotation_boxes[:, 1] = _annotation_boxes[:, 1]
            annotation_boxes[:, 2] = _annotation_boxes[:, 2]
            annotation_boxes[:, 3] = _annotation_boxes[:, 3] 
#            
            for label in range(cfg['num_classes']):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]
#
average_precisions = {}
for label in range(cfg['num_classes']):
    true_positives = []
    scores = []
    num_annotations = 0  ###  total num of object label
    
    for i in range(len(all_annotations)):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]
        num_annotations += annotations.shape[0]
        detected_annotations = []
        
        for *bbox, score in detections:
            scores.append(score)
            if annotations.shape[0] == 0:
                true_positives.append(0)
                continue
            
            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)
            
    # no annotations -> AP for this class is 0
    if num_annotations == 0:
        average_precisions[label] = 0
        continue
    true_positives = np.array(true_positives)
    false_positives = np.ones_like(true_positives) - true_positives
    
    # sort by score
    indices = np.argsort(-np.array(scores))   #从大到小
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]
##
    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)
##
    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision
  
 
print("Average Precisions:")
for c, ap in average_precisions.items():
    print(f"+ Class '{c}' - AP: {ap}")

mAP = np.mean(list(average_precisions.values()))
print(f"mAP: {mAP}")
     
    