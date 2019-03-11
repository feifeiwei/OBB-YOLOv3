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
              'img_shape': 416,
              'train_path': r'E:\遥感车辆数据集\DOTA\train\DOTA_train_order2.txt',
              'test_path': r'E:\遥感车辆数据集\UCAS\test.txt',
              'anchors' :  [[(85,44), (84,89), (49,72)],
                           [(45,34), (40,21), (34,42)],
                           [(23,40), (19,22), (9,10)]] ,
              'num_classes': 2,
              'name_path' : ["car, big_car"],
              'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]],
              'save_name':'dota_ckpt.pth'
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
