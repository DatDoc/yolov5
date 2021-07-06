# Generate submission file for VinDr-CXR
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse

def filter_2cls(row, low_thr=0.06, high_thr=0.95):
    prob = row['target']
    if prob<low_thr:
        ## Less chance of having any disease
        row['PredictionString'] = '14 1 0 0 1 1'
    elif low_thr<=prob<high_thr:
        ## More change of having any diesease
        row['PredictionString']+=f' 14 {prob} 0 0 1 1'
    elif high_thr<=prob:
        ## Good chance of having any disease so believe in object detection model
        row['PredictionString'] = row['PredictionString']
    else:
        raise ValueError('Prediction must be from [0-1]')
    return row

def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes

def gen_sub(opt):
    test_df = pd.read_csv(opt.test_csv)
    image_ids = []
    PredictionStrings = []
    
    for file_name in tqdm(os.listdir(opt.txt_path)):
        
        image_id = file_name.split('.')[0]
        file_path = os.path.join(opt.txt_path, file_name)

        w, h = test_df.loc[test_df.image_id==image_id,['width', 'height']].values[0]
        f = open(file_path, 'r')
        data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)
        data = data[:, [0, 5, 1, 2, 3, 4]]
        bboxes = list(np.round(np.concatenate((data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis =1).reshape(-1), 1).astype(str))
        for idx in range(len(bboxes)):
            bboxes[idx] = str(int(float(bboxes[idx]))) if idx%6!=1 else bboxes[idx]
        image_ids.append(image_id)
        PredictionStrings.append(' '.join(bboxes))


    pred_df = pd.DataFrame({'image_id':image_ids,
                            'PredictionString':PredictionStrings})
    sub_df = pd.merge(test_df, pred_df, on = 'image_id', how = 'left').fillna("14 1 0 0 1 1")
    sub_df = sub_df[['image_id', 'PredictionString']]
    if opt.cls2 is not None:
        pred_14cls = sub_df
        pred_2cls = pd.read_csv(opt.cls2)
        pred = pd.merge(pred_14cls, pred_2cls, on = 'image_id', how = 'left')
        sub_df = pred.apply(filter_2cls, axis=1)
        sub_df = sub_df[['image_id', 'PredictionString']]

    sub_df.to_csv(os.path.join(opt.dst, opt.name + '.csv'),index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, help='test.csv')
    parser.add_argument('--txt_path', type=str, help='inference labels path')
    parser.add_argument('--dst', type=str, help='path to save file')
    parser.add_argument('--name', type=str, help='submission file name')
    parser.add_argument('--cls2', type=str, help='2 class filter')
    opt = parser.parse_args()
    gen_sub(opt)