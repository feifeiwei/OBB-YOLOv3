# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:11:07 2018

@author: 60236
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .augmentation import random_flip, random_flip_updown, resize, random_crop
import cv2

    
class ImageFolder(Dataset):
    def __init__(self, root, folder_path, img_size=416,transform=None):
        self.img_shape = (img_size, img_size)
        self.transform = transform
        
        with open(folder_path,'r') as f:
            files = f.readlines()
            self.num_samples = len(files)
        files = [i.strip() for i in files]
        self.img_files = [os.path.join(root,i.split(' ')[0]) for i in files]
        
    def __getitem__(self,index):
        image_path = self.img_files[index % self.num_samples]
        #extract images
        img = np.array(Image.open(image_path))  # h w 
        # Black and white images
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if img.shape[2]==4:
            img = img[:,:,:3]
        
        h, w ,_ = img.shape
        dim_diff = np.abs(h-w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff // 2
        #Determine padding
        
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128)
        # Resize and normalize
        input_img = cv2.resize(input_img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        if self.transform is not None:
            input_img = self.transform(input_img)
        else:
            input_img = np.transpose(input_img, (2, 0, 1)) / 255.
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()
        return image_path, input_img
    def __len__(self):
        return self.num_samples
    
    
class ListDataset(Dataset):
    '''
        input: im_name pieces x1 y1 x2 y2 x3 y3 x4 y4 pieces......
        return:
            im_name, input_img, filled_labels[x1 y1 x2 y2 x3 y3 x4 y4 pieces...]
    '''
    def __init__(self,root, list_path, img_size=416, transform=None, train=True):
        with open(list_path,'r') as f:
            files = f.readlines()
            self.num_samples = len(files)
        files = [i.strip() for i in files]
        self.img_files = [os.path.join(root,i.split(' ')[0]) for i in files]
        self.label_files = [i.split(' ')[1:] for i in files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 200
        self.transform = transform
        self.train = train
        #self.root = root
        
    def __getitem__(self,index):
        #-----------
        #image
        #-----------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)

        img = np.array(img)
        # Black and white images
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        img = img[...,:3]
          
        ##########################################################
        #---------
        #  Label
        #---------
        label_path = self.label_files[index % len(self.img_files)]
        label_path = [float(i) for i in label_path]
        labels = np.array(label_path).reshape(-1,9).astype('float64')
         # Extract coordinates for unpadded + unscaled image
        x1 = labels[:,0]
        y1 = labels[:,1]
        x2 = labels[:,2]
        y2 = labels[:,3]
        x3 = labels[:,4]
        y3 = labels[:,5]
        x4 = labels[:,6]
        y4 = labels[:,7]
        c = labels[:,8]

#        x1,y1,x2,y2 = x1/ratio_, y1/ratio_, x2/ratio_, y2/ratio_
        
        boxes = np.zeros((x1.shape[0],11))
        boxes[:,0] = x1
        boxes[:,1] = y1
        boxes[:,2] = x2
        boxes[:,3] = y2
        boxes[:,4] = x3
        boxes[:,5] = y3
        boxes[:,6] = x4
        boxes[:,7] = y4

        boxes[:,8] = boxes[:,[0,2,4,6]].sum(1) / 4.
        boxes[:,9] = boxes[:,[1,3,5,7]].sum(1) / 4.
        boxes[:,10] = c
        #----------------------------------------------------------------------
        #  data Augmentation
        #--------------------------------------------------------------------
        input_img = Image.fromarray(img)
        if self.train == True:
            input_img, boxes = random_flip(input_img, boxes)
            input_img, boxes = random_flip_updown(input_img, boxes)
            input_img, boxes = random_crop(input_img, boxes)
            input_img, boxes = resize(input_img, boxes,self.img_shape)
            
#        
#        
#        #---------------------------------------------------------------------------
         #Calculate ratios from coordinates
        boxes[:,0] /= self.img_shape[0]
        boxes[:,1] /= self.img_shape[0]
        boxes[:,2] /= self.img_shape[0]
        boxes[:,3] /= self.img_shape[0]
        boxes[:,4] /= self.img_shape[0]
        boxes[:,5] /= self.img_shape[0]
        boxes[:,6] /= self.img_shape[0]
        boxes[:,7] /= self.img_shape[0]
        
        boxes[:,8] /= self.img_shape[0]
        boxes[:,9] /= self.img_shape[0]
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 11))
        if boxes is not None:          
            filled_labels[range(len(boxes))] = boxes[:self.max_objects]

        if self.transform is None:
            #Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))/255.
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()
        else:
            input_img = self.transform(input_img) 
        return img_path, input_img, filled_labels
    def __len__(self):
        return (self.num_samples)
        
        
        
        
if __name__=="__main__":
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    #from model import yolov3
    from config import ucas_config


    root = r'/home/ubantu/datasets/DOTA/train/images'
    ii = r'/home/ubantu/datasets/DOTA/train_order.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
    transforms.ToTensor(),
#    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
    
    
    img_size = 416
   # model = yolov3(img_size,ucas_config['anchors'],1).cuda()
    da = ListDataset(root, ii,transform=transform,img_size=img_size)
    dataloader = torch.utils.data.DataLoader(da,batch_size=1,shuffle=1)
    
# 
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
      
#        #loss = model(imgs.cuda(),targets.cuda())
##        break
        imgs = imgs.squeeze(0).permute(1,2,0).numpy().copy()
        for i in range(80):
#            if i==2:
#                break
            if targets[0,i].sum() == 0:
                break
            
            label = targets[0][i][0:-3].data.cpu().numpy()*img_size #中心坐标 + 宽高
            center = targets[0][i][-3:-1].data.cpu()*img_size
            
            
        
            coor = label.reshape((-1,1,2)).astype(np.int32)
            point = tuple((center.numpy().astype(np.int32)))
            
            imgs = cv2.polylines(imgs, [coor],True, (0,0,255),thickness=4)
            imgs = cv2.circle(imgs,point,1,(0,255,0),4)
        plt.imshow(imgs)
        
        break
        
        
        
        