# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:11:07 2018
@author: 60236
"""
import os
import argparse

import torch
import torch.nn as nn
from config import *
import torch.optim as optim
from module.model import yolov3

import torch.backends.cudnn as cudnn
from datasets.datasets import ListDataset
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=27, help='number of epochs')
parser.add_argument('--lr', type=int, default=1e-3, help='learing rate for training')
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()
print(opt)

batch_size = opt.batch_size
learning_rate = opt.lr
epochs = opt.epochs
cfg = Dota_config


os.makedirs('output',exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0]

model = yolov3(cfg['img_shape'], cfg['anchors'], cfg['num_classes'])
optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9,weight_decay=5e-4)

if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoints/'+cfg['save_name'])
    model.load_state_dict(checkpoint['weights'])
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
else:
    print('initial model from DarkNet..')
    start_epoch = 0
    #model.load_weights('./checkpoints/darkNet53.pth')
    best_loss = float('inf')
    

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
    
    
dataloader = torch.utils.data.DataLoader(
        ListDataset(cfg['root'],cfg['train_path'],img_size=cfg['img_shape'],
                    transform=transform, train=True),
        batch_size=batch_size,
        shuffle=True,)



testloader = torch.utils.data.DataLoader(
        ListDataset(cfg['root'], cfg['test_path'],img_size=cfg['img_shape'],
                    transform=transform, train=False),
        batch_size=8,shuffle=False
        )
print('training samples are %d'%(batch_size*len(dataloader)))
print('testing samples are %d'%(8*len(testloader)))
        


#### mult   GPUs
if len(device_ids) > 1: 
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
else:
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])
    model = model.to(device)


print("=> start training")
for epoch in range(start_epoch,start_epoch+epochs):
    model.train()
    cur_loss = 0
    for i, (_,imgs,targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        losses = model(imgs,targets)  #[loss,x,y,w,h,conf,cls]
        loss = losses[0]
        loss.backward()
        if len(device_ids) > 1:
            optimizer.module.step()
        else:
            optimizer.step()
        
        print('[Epoch %d/%d, Batch %d/%d] [x:%.3f, y:%.3f, x1:%.3f, y1:%.3f, x2:%.3f, y2:%.3f, x3:%.3f, y3:%.3f, x4:%.3f, y4:%.3f, conf:%.3f, cls:%.3f, total:%.3f]' %
                                    (epoch, opt.epochs, i, len(dataloader),
                                    losses[1], losses[2], losses[3],
                                    losses[4], losses[5], losses[6],
                                    losses[7], losses[8], losses[9],
                                    losses[10], losses[11],losses[12],
                                    loss.item()))
        
#    model.eval()
#    print("Testing...")
#    for i, (_,imgs,targets) in enumerate(testloader):
#        imgs = imgs.to(device)
#        targets = targets.to(device)
#        optimizer.zero_grad()
#        
#        losses = model(imgs,targets)  #[loss,x,y,w,h,conf,cls]
#        loss = losses[0]
#        print('[Epoch %d/%d, Batch %d/%d] [x:%.3f, y:%.3f, x1:%.3f, y1:%.3f, x2:%.3f, y2:%.3f, x3:%.3f, y3:%.3f, x4:%.3f, y4:%.3f, conf:%.3f, cls:%.3f, total:%.3f]' %
#                                    (epoch, opt.epochs, i, len(testloader),
#                                    losses[1], losses[2], losses[3],
#                                    losses[4], losses[5], losses[6],
#                                    losses[7], losses[8], losses[9],
#                                    losses[10], losses[11],losses[12],
#                                    loss.item()))
        cur_loss += loss.item()
    
    cur_loss /= (i+1)
    if cur_loss < best_loss:
        print("\nModel saving ...  |   The cur loss is: ", cur_loss)
        print("\n")
        if len(device_ids) > 1:
            weight = model.module.state_dict()
        else:
            weight = model.state_dict()
        state = {
              "weights": weight,#,#
              "best_loss": cur_loss,
              "epoch": epoch
              }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(state, './checkpoints/%s'%(cfg["save_name"]))
        best_loss = cur_loss
        
        
        
    