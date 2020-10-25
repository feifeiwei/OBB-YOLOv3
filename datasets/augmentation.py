# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:54:20 2018
数据增强 data
@author: 60236
"""
import math
import torch
import random
import numpy as np
from .utils import get_best_begin_point
from PIL import Image, ImageDraw

def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,8].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    
    boxes = torch.from_numpy(boxes).float()
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    img = img.resize((ow,oh), Image.BILINEAR)
    #import pdb
    #pdb.set_trace()
    boxes[:,:-1] = boxes[:,:-1]*torch.Tensor([sw,sh,sw,sh,sw,sh,sw,sh,sw,sh])
    tmp_boxes = boxes[:,:8]
    
    for i, box in enumerate(tmp_boxes):
        box = box.reshape(-1,2)
        res = get_best_begin_point(box)
        res = torch.from_numpy(np.array(res).reshape(1,-1)[0])
        tmp_boxes[i] = res
    boxes[:,8] = tmp_boxes[:,[0,2,4,6]].sum(1) / 4.
    boxes[:,9] = tmp_boxes[:,[1,3,5,7]].sum(1) / 4.
    boxes[:,:8] = tmp_boxes
    
    return img, boxes
           



def random_flip(img, boxes):
    '''Randomly flip the given PIL Image.
        
    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb, 11].

    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 1:
        
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        x24 = w - boxes[:,[0,2,4,6]]
       
        boxes[:,[0,2,4,6]] = x24
    tmp_boxes = boxes[:,:8]
    
    for i, box in enumerate(tmp_boxes):
        box = box.reshape(-1,2)
        res = get_best_begin_point(box)
        res = np.array(res).reshape(1,-1)[0]
        tmp_boxes[i] = res
    boxes[:,8] = tmp_boxes[:,[0,2,4,6]].sum(1) / 4.
    boxes[:,9] = tmp_boxes[:,[1,3,5,7]].sum(1) / 4.
    boxes[:,:8] = tmp_boxes
    return img, boxes       

def random_flip_updown(img, boxes):
    '''Randomly flip the given PIL Image.
        
    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        h = img.height
        boxes[:,[1,3,5,7]] = h - boxes[:,[1,3,5,7]]
    tmp_boxes = boxes[:,:8]
    for i, box in enumerate(tmp_boxes):
        box = box.reshape(-1,2)
        res = get_best_begin_point(box)
        res = np.array(res).reshape(1,-1)[0]
        tmp_boxes[i] = res
    boxes[:,8] = tmp_boxes[:,[0,2,4,6]].sum(1) / 4.
    boxes[:,9] = tmp_boxes[:,[1,3,5,7]].sum(1) / 4.
    boxes[:,:8] = tmp_boxes
    return img, boxes   

    
def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.
    A crop of random size of (0.8 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,8].
    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.9, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2
    img = img.crop((x, y, x+w, y+h))
    boxes[:,:8] -= np.array([x,y,x,y,x,y,x,y])
    boxes[:,[0,2,4,6]] = boxes[:,[0,2,4,6]].clip(min=0, max=w-1)
    boxes[:,[1,3,5,7]] = boxes[:,[1,3,5,7]].clip(min=0, max=h-1)
    
    boxes[:,8] = boxes[:,[0,2,4,6]].sum(1) / 4.
    boxes[:,9] = boxes[:,[1,3,5,7]].sum(1) / 4.
    
    return img, boxes

    
    
if __name__=="__main__":
    import cv2
    
    import matplotlib.pyplot as plt
    img = Image.open(r'E:\遥感数据集\DOTA\split\train\images\P0000_0.png')
    
    box = '285 606 391 542 466 669 371 729 514 532 627 468 701 592 588 659'
    box = list(map(int, box.split(' ')))
    box = np.array(box).reshape(-1,8)
#    img, box = random_flip_updown(img, box)
    img, box = random_crop(img, box)
    img = np.array(img)
    for i in box:
        box1 = i.reshape((-1,1,2))
    
        img = cv2.polylines(img, [box1],True,(0,255,0), thickness=3)
    plt.imshow(img)
    plt.show()
    
