# -*- coding: utf-8 -*-
import torch
import numpy as np

def predict_transform(fms, inp_dim, anchors, num_classes, cuda=False):
    '''
        ?x255x13x13,26x26,52x52    3*(11+80)=255
    '''
    stride =  inp_dim // fms.size(2)  #416// 13,26,52 = 32, 6, 8
    batch_size = fms.size(0)
    bbox_attrs = 11 + num_classes                ##5+80 = 85
    grid_size = inp_dim // stride               ###   13,26,52
    #[(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    num_anchors = len(anchors)                   # 3   1
    # [?,255,169]///676,2704
    prediction = fms.view(batch_size, bbox_attrs*num_anchors,-1)
    ##[?,169,255]
    prediction = prediction.transpose(1,2).contiguous()
    
    #[?,169*3,85]
    prediction = prediction.view(batch_size, -1, bbox_attrs)   #?, 507,31
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce  ##0-7坐标 ； 8，9 中心  10 conf
    prediction[:,:,8] = torch.sigmoid(prediction[:,:,8])  #?,507
    prediction[:,:,9] = torch.sigmoid(prediction[:,:,9])
    prediction[:,:,10] = torch.sigmoid(prediction[:,:,10])
    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len,grid_len)  ##16*16,  16*16
    
    x_offset = torch.FloatTensor(a).view(-1,1) #0,1,2,3,...15..0,1,2
    y_offset = torch.FloatTensor(b).view(-1,1) #0,0,0,0,0,0,...1,1,1,1...15,15,15
    #[1,507,2]  ---> 2028, 8112
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)  #1,768,2
   
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors).repeat(1,4)#[1.25,1.625],[2,3.75],[4.125,2.875]
    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)##[507,2]->[1,507,2]
    
    if cuda:
        x_y_offset = x_y_offset.cuda()
        anchors = anchors.cuda()
    prediction[...,8:10] += x_y_offset  #加上偏置  【?, 507, 2】
#    prediction[...,0:8] = torch.exp(prediction[:,:,0:8])*anchors
    prediction[...,0:8] = prediction[:,:,0:8] * anchors + x_y_offset.repeat(1,1,4)
    #Softmax the class scores
    prediction[...,11: 11 + num_classes] = torch.sigmoid((prediction[:,:, 11 : 11 + num_classes]))
    prediction[...,:10] *= stride 
    
    return prediction


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def get_target( target, anchors, g_dim, ignore_threshold, num_classes):
    '''
        target:   ?, 50,11
        anchors: scaled anchors.   / stride
        g_dim:  feature map size 13,26,52
        
        return :
            
    
    '''
    bs = target.size(0)
    nA = len(anchors)
    num_classes = num_classes
    
    mask = torch.zeros(bs, nA, g_dim, g_dim)
    conf_mask = torch.ones(bs, nA, g_dim, g_dim)
    
    tx = torch.zeros(bs, nA, g_dim, g_dim)
    ty = torch.zeros(bs, nA, g_dim, g_dim)
    tx1 = torch.zeros(bs, nA, g_dim, g_dim)
    ty1 = torch.zeros(bs, nA, g_dim, g_dim)
    tx2 = torch.zeros(bs, nA, g_dim, g_dim)
    ty2 = torch.zeros(bs, nA, g_dim, g_dim)
    tx3 = torch.zeros(bs, nA, g_dim, g_dim)
    ty3 = torch.zeros(bs, nA, g_dim, g_dim)
    tx4 = torch.zeros(bs, nA, g_dim, g_dim)
    ty4 = torch.zeros(bs, nA, g_dim, g_dim)
    tconf = torch.zeros(bs, nA, g_dim, g_dim)
    tcls = torch.zeros(bs, nA, g_dim, g_dim, num_classes)
    #tcls = torch.zeros(bs, nA, g_dim, g_dim)
    
    for b in range(bs):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                break
            # Convert to position relative to box
            gx1 = target[b, t, 0] * g_dim
            gy1 = target[b, t, 1] * g_dim
            gx2 = target[b, t, 2] * g_dim
            gy2 = target[b, t, 3] * g_dim
            gx3 = target[b, t, 4] * g_dim
            gy3 = target[b, t, 5] * g_dim
            gx4 = target[b, t, 6] * g_dim
            gy4 = target[b, t, 7] * g_dim
            gx = target[b, t, 8] * g_dim
            gy = target[b, t, 9] * g_dim
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            
            gw = max(target[b,t,[0,2,4,6]] * g_dim) - min(target[b,t,[0,2,4,6]] * g_dim)
            gh = max(target[b,t,[1,3,5,7]] * g_dim) - min(target[b,t,[1,3,5,7]] * g_dim)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((nA, 2)),
                                                                  np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_threshold, gj, gi] = 0

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            
            tx1[b, best_n, gj, gi] = (gx1 - gi) / anchors[best_n][0]
            ty1[b, best_n, gj, gi] = (gy1 - gj) / anchors[best_n][1]
            tx2[b, best_n, gj, gi] = (gx2 - gi) / anchors[best_n][0]
            ty2[b, best_n, gj, gi] = (gy2 - gj) / anchors[best_n][1]
            tx3[b, best_n, gj, gi] = (gx3 - gi) / anchors[best_n][0]
            ty3[b, best_n, gj, gi] = (gy3 - gj) / anchors[best_n][1]
            tx4[b, best_n, gj, gi] = (gx4 - gi) / anchors[best_n][0]
            ty4[b, best_n, gj, gi] = (gy4 - gj) / anchors[best_n][1]
            
            
            
            # object
            tconf[b, best_n, gj, gi] = 1
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1
            #tcls[b, best_n, gj, gi] = int(target[b, t, 4])
            
    return mask, conf_mask, tx, ty, tx1, ty1, tx2, ty2,tx3, ty3,tx4, ty4, tconf, tcls   