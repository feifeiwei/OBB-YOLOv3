# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:49:30 2018

@author: 60236
"""

import numpy as np
import torch
import cv2


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
      boxes: (tensor) bounding boxes, sized [N,8]

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy','xiyi2xyxy']
    
    if order == 'xyxy2xywh':
        a = boxes[:,:2]
        b = boxes[:,2:]
        return torch.cat([(a+b)/2,b-a+1], 1)
    elif order == 'xywh2xyxy':
        a = boxes[:,:2]
        b = boxes[:,2:]
        return torch.cat([a-b/2,a+b/2], 1)
    else:
        num_ = boxes.shape[0]
        tmp = torch.zeros(num_,4)
        for i in range(num_):
            cur_8bit = boxes[i].cpu().numpy().reshape(-1,2)
            boxes_xywh = cv2.boundingRect(cur_8bit)
#            boxes_xywh = torch.from_numpy(boxes_xywh)
            tmp[i][0] = boxes_xywh[0]
            tmp[i][1] = boxes_xywh[1]
            tmp[i][2] = boxes_xywh[0] + boxes_xywh[2]
            tmp[i][3] = boxes_xywh[1] + boxes_xywh[3]
        return tmp

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
    
def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
       ?, 10647,13      0-7 coors ;  8-9 center coor; 10 conf;  11-> classes;
    
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (xi, yi) i=4 to (x1, y1, x2, y2)
#    shape = (prediction.shape[0], prediction.shape[1], 4)
#    box_4bit = torch.zeros(shape)
#    for i in range(prediction.shape[0]):
#        
#        box_4bit[i] = change_box_order(prediction[i][:, :8], order='xiyi2xyxy')
#        
#    prediction = torch.cat((prediction, box_4bit.cuda()), 2)  ###14:18  4bit boxes
    
    
    output = [None] * len(prediction)
    
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 10] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
#        
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 10:10 + num_classes], 1,  keepdim=True)
        # Detections ordered as (xi, yi, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, [0,1,2,3,4,5,6,7,10]], class_conf.float(), class_pred.float()), 1)
#        
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()    
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 8], descending=True)
            detections_class = detections_class[conf_sort_index]  ###?, 10
            boxes_4bit = change_box_order(detections_class[:, :8], order='xiyi2xyxy')
            if detections_class.is_cuda:
                boxes_4bit = boxes_4bit.cuda()
            detections_class = torch.cat((detections_class, boxes_4bit), 1)  ####  0-7 8bit coors;   8 obj_conf; 9 class_conf; 10class_pred; 11-14: 4bit coor
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1][:, 11:], detections_class[1:, 11:])
                #print(max_detections[-1][11:], '***', detections_class[1:, 11:])
             
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]
               
            max_detections = torch.cat(max_detections).data
                #Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    return output
        


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap