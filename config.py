# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:11:27 2019

@author: 60236
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:19:54 2018

@author: 60236
"""

import os
import os.path

pwd = os.getcwd()
VOCroot = os.path.join(pwd, "data/datasets/VOCdevkit0712/")
COCOroot = os.path.join(pwd, "data/datasets/coco2015")

datasets_dict = {"VOC": [('0712', '0712_trainval')],
            "VOC0712++": [('0712', '0712_trainval_test')],
            "VOC2012" : [('2012', '2012_trainval')],
            "COCO": [('2014', 'train'), ('2014', 'valminusminival')],
            "VOC2007": [('0712', "2007_test")],
            "COCOval": [('2014', 'minival')]}


voc_config = {
    'anchors' : [[(166, 230), (272, 148), (309, 274)], ###voc
                [(62, 47), (97, 176), (132, 97)], 
                [(24, 32), (32, 71), (59, 106)]],
    'root': VOCroot,
    'num_classes': 20,
    'multiscale' : True,
    'name_path' : './data/voc_classes.txt',
    'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]]
}

ucas_config = {
              'img_shape': 416,
              'train_path': r'E:\遥感车辆数据集\UCAS\train_ubuntu.txt',
              'test_path': r'E:\遥感车辆数据集\UCAS\test.txt',
              'anchors' :  [[(81,44), (71,36), (64,57)],
                           [(61,40), (58,33), (46,54)],
                           [(45,78), (38,68), (33,55)]] ,
              'num_classes': 1,
              'name_path' : "./data/ucas_classes.txt",
              'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]],
              'save_name':'ucas_ckpt.pth'
}

Dota_config = {
              'img_shape': 416, #1120
              'train_path': r'/home/weimf/car_datasets/DOTA/train/DOTA_train_order_ubuntu.txt',
              'test_path': r'E:\遥感车辆数据集\DOTA\val\DOTA_val_order.txt',
              'demo_path':r'C:\Users\60236\Desktop\dota\0\demo.txt',
              'anchors' :  [[(85,44), (84,89), (49,72)],
                           [(45,34), (40,21), (34,42)],
                           [(23,40), (19,22), (9,10)]] ,
              'num_classes': 2,
              'name_path' : ["car, big_car"],
              'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]],
              'save_name':'dota_ckpt6.pth'
        }
UCAS_8bit_config = {
              'img_shape': 1280,
              'train_path': r'/home/share/UCAS/UCAS_8bit.txt',
              'test_path': r'/home/share/UCAS/UCAS_8bit_test.txt',
              'anchors' :  [[(81,44), (71,36), (65,57)],
                           [(61,40), (58,33), (46,54)],
                           [(45,78), (38,68), (33,55)]],
              'num_classes': 1,
              'name_path' : ["car"],
              'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]],
              'save_name':'UCAS_8bit_ckpt.pth'
        }
        
        
        
coco_config = {
    'anchors' : [[(116, 90), (156, 198), (373, 326)], 
                [(30, 61), (62, 45), (59, 119)], 
                [(10, 13), (16, 30), (33, 23)]],
    'root': COCOroot,
    'num_classes': 80,
    'multiscale' : True,
    'name_path' : "./data/coco.names",
    'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]]
}

nwpu_config = {
              'img_shape': 800,
              'anchors' :  [[(349,456), (495,278), (597,463)],
                           [(94,83), (123,434), (234,253)],
                           [(29,27), (39,41), (61,57)]] ,
              'num_classes': 10,
              'multiscale' : True,
              'name_path' : "./data/nwpu_classes.txt",
              'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]]
}
