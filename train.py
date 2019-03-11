# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:55:50 2019
@author: 60236
"""
import os
import argparse

import torch
import torch.nn as nn
from config import *
import torch.optim as optim
from module.model import yolov3

from datasets.datasets import ListDataset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=int, default=1e-3, help='learing rate for training')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

batch_size = opt.batch_size
learning_rate = opt.lr
epochs = opt.epochs
cfg = Dota_config


os.makedirs('output',exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids=[0,3]


model = yolov3(cfg['img_shape'], cfg['anchors'], cfg['num_classes'])
cudnn.benchmark = True
model = torch.nn.DataParallel(model,device_ids=device_ids).to(device)

if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoints/'+cfg['save_name'])
    model.load_state_dict(checkpoint['weights'])
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
else:
    print('initial model from DarkNet..')
    start_epoch = 0
    model.load_weights('./checkpoints/darkNet53.pth')
    best_loss = float('inf')
# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
    
    
dataloader = torch.utils.data.DataLoader(
        ListDataset(cfg['train_path'],img_size=cfg['img_shape'],
                    transform=transform, train=True),
        batch_size=batch_size,
        shuffle=True,)

testloader = torch.utils.data.DataLoader(
        ListDataset(cfg['test_path'],img_size=cfg['img_shape'],
                    transform=transform, train=False),
        batch_size=16,
        shuffle=False,)


print('training samples is %d'%(batch_size*len(dataloader)))
print('testing samples is %d'%(16*len(dataloader)))


optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9,weight_decay=5e-4)
optimizer = nn.DataParallel(optimizer, device_ids=device_ids)



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
        optimizer.module.step()
        
        print('[Epoch %d/%d, Batch %d/%d] [x: %f, y: %f, x1: %f, y1: %f, x2: %f, y2: %f, x3: %f, y3: %f, x4: %f, y4: %f, conf: %f, cls: %f, total: %f]' %
                                    (epoch, opt.epochs, i, len(dataloader),
                                    losses[1], losses[2], losses[3],
                                    losses[4], losses[5], losses[6],
                                    losses[7], losses[8], losses[9],
                                    losses[10], losses[11],losses[12],
                                    loss.item()))
        
#        cur_loss += loss.item()
#    
#    cur_loss /= i
#    
#    if cur_loss < best_loss:
#        print('\nSaving ....  | the val loss is: ',cur_loss)
#        print('\n')
#        state = {
#                 'weights':    model.state_dict(),
#                 'best_loss':       cur_loss,
#                 'epoch':      epoch,
#                }
#        if not os.path.isdir('checkpoints'):
#            os.mkdir('checkpoints')
#        torch.save(state,'./checkpoints/ckpt.pth')
#        best_loss = cur_loss
    model.eval()
    for i, (_, imgs, targets) in enumerate(testloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        losses = model(imgs,targets)  #[loss,x,y,w,h,conf,cls]
        loss = losses[0]
        print('[Epoch %d/%d, Batch %d/%d] [x: %f, y: %f, x1: %f, y1: %f, x2: %f, y2: %f, x3: %f, y3: %f, x4: %f, y4: %f, conf: %f, cls: %f, total: %f]' %
                                    (epoch, opt.epochs, i, len(testloader),
                                    losses[1], losses[2], losses[3],
                                    losses[4], losses[5], losses[6],
                                    losses[7], losses[8], losses[9],
                                    losses[10], losses[11],losses[12],
                                    loss.item()))
        cur_loss += loss.item()
    if cur_loss < best_loss:
        print('\nSaving ....  | the val loss is: ',cur_loss)
        print('\n')
        state = {
                 'weights':    model.module.state_dict(),
                 'best_loss':       cur_loss,
                 'epoch':      epoch,
                }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state,'./checkpoints/ckpt.pth')
        best_loss = cur_loss
        
    