"""
Created on Aug 27
@author: STRH
"""

# import libraries
import cv2
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import os.path as osp
from pathlib import Path

# import tracker
from Tracker import Tracker
from model.AbaViTrack import AbaViTrack
from model.head import CenterPredictor
from model.AbaViT import abavit_patch16_224

def parse_args():
    parser = argparse.ArgumentParser(description="SiamBAN Inference to Evaluation")
    parser.add_argument('--data_path', type=str, required=True, help="Path to input images folder or video file")
    parser.add_argument('--save_path', type=str, required=False, help="path to folder for saving results")
    parser.add_argument('--update_rate', type=int, required=True, help="Update rate")
    parser.add_argument('--weights', type=str, default='checkpoints/ckpt.pth')
    return parser.parse_args()

def build_box_head(in_channel, out_channel, search_size, stride):
    feat_sz = search_size / stride
    center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                  feat_sz=feat_sz, stride=stride)
    return center_head


def build_model():
    search_size = 256
    stride = 16
    backbone = abavit_patch16_224()
    box_head = build_box_head(backbone.embed_dim, 256, search_size, stride)

    model = AbaViTrack(
        backbone,
        box_head
    )

    return model

def get_ground_truth(path):   
    file = open(path, "r")
    gTruth = []
    for item in file:
        item = item.strip()det
        item = item.strip(";")
        item = eval(item)
        gTruth.append(item)
    return gTruth

def groundTruth(path):
    gTruth = get_ground_truth(path)
    for j in range(0,len(gTruth)):
        if len(gTruth[j])!=0:
            gTruth[j] = gTruth[j][1:5]
            
    return gTruth


def main():
    args = parse_args()
    model = build_model()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=False)

    tracker = Tracker(model)

    # creating save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Save path created!")

    vid_path = args.data_path
    output_file = args.save_path + f"/{vid_path.split('/')[-1].split('.')[0]}.txt"

    vid = cv2.VideoCapture(vid_path)
    
    frame_id = 0
    i = 0

    gt_path = f"{args.data_path.split('.')[0]}.txt"
    gTruth = groundTruth(path=gt_path)
    
    with open(output_file, 'w') as f:
        while True:
            check = False
            results = []
            ret, img = vid.read()
            if ret==0:
                break

            # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
            # dets = np.array([[584.000,382.000,584.000 + 81.000,32.000 + 382.000, 0.9, 0]])
            t1 = time.time()

            # using another loading method
            det = gTruth[frame_id]
            if len(det)>0:
                i+=1
            
            if i==1:
                x,y,w,h = det
                tracker.initialize(img, (x,y,w,h))
                gt = True

            if frame_id % args.update_rate == 0:
                check = True

            if check:
                if len(det)!= 0:
                    x,y,w,h = det
                    tracker.initialize(img, (x,y,w,h))
                    gt = True

                else:
                    gt = False

            if gt:
                out = tracker.track(img)
                x,y,w,h = out
                x2, y2 = x + w, y + h
                
                # results.append([0,x1,y1,x2,y2,output['best_score']])
                results.append([0,x,y,x2,y2,1])

                img = cv2.rectangle(img, (int(x),int(y)), (int(x2),int(y2)), (0,255,255))
                cv2.putText(img,f"Quad, TraDet Score: {1}",(int(x),int(y)-15),0,1,(192,227,16))

            f.write(str(results)+"\n")
                    
            # break on pressing q or space
            cv2.imshow('SiamFC Personal Demo:', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
            frame_id += 1
    f.close()

    cv2.destroyAllWindows()
    vid.release()

    print(f"Tracking results saved to {output_file}")

if __name__ == "__main__":
    main()