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




Dota_config = {
              'img_shape': 416, #1120
              'root':r'/home/ubantu/datasets/DOTA/train/images',
              'test_root':r'/home/ubantu/datasets/DOTA/val/images',
              'train_path': r'/home/ubantu/datasets/DOTA/train_order.txt',
              'test_path': r'/home/ubantu/datasets/DOTA/val_order.txt',
              'demo_path':r'C:\Users\60236\Desktop\dota\0\demo.txt',
              'anchors' :  [[(85,44), (84,89), (49,72)],
                           [(45,34), (40,21), (34,42)],
                           [(23,40), (19,22), (9,10)]] ,
              'num_classes': 2,
              'name_path' : ["car, big_car"],
              'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]],
              'save_name':'dota_ckpt6.pth'
        }