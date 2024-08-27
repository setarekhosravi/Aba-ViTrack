import argparse
import glob
import os
import time
from typing import List

import numpy as np
import torch
import cv2 as cv
from Tracker import Tracker
from model.AbaViTrack import AbaViTrack
from model.head import CenterPredictor
from model.AbaViT import abavit_patch16_224


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


def read_image(image_file: str):
    if isinstance(image_file, str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    else:
        raise ValueError("type of image_file should be str or list")


def save_bb(file, data):
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')


def save_time(file, data):
    exec_times = np.array(data).astype(float)
    np.savetxt(file, exec_times, delimiter='\t', fmt='%f')


def main():
    parser = argparse.ArgumentParser(description='Run a tracking demo.')
    parser.add_argument('--initial_bbox', nargs='+', type=int, required=True) # 499 421 102 179
    parser.add_argument('--frames_path', type=str, default='frames')
    parser.add_argument('--weights', type=str, default='checkpoints/ckpt.pth')
    parser.add_argument('--output_path', type=str, default='outputs')
    parser.add_argument('--bbox_file', type=str, default='bbox.txt')
    parser.add_argument('--time_file', type=str, default='time.txt')

    args = parser.parse_args()

    sequence_list = sorted(glob.glob(os.path.join(args.frames_path, '*.jpg')))

    model = build_model()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=False)

    tracker = Tracker(model)

    pred_box = []
    times = []

    image = read_image(sequence_list[0])
    pred_box.append(args.initial_bbox)
    tracker.initialize(image, args.initial_bbox)
    for frame_num, frame_path in enumerate(sequence_list[1:], start=1):
        image = read_image(frame_path)
        start_time = time.time()
        out = tracker.track(image)
        x,y,w,h = out
        cv.rectangle(image, (int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),3)
        cv.imshow('show',image)
        times.append(time.time() - start_time)
        pred_box.append(out)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path, mode=0o777)

    bbox_file = os.path.join(args.output_path, args.bbox_file)
    time_file = os.path.join(args.output_path, args.time_file)
    save_bb(bbox_file, pred_box)
    save_time(time_file, times)
    print("FPS: ", len(times)/sum(times))


if __name__ == '__main__':
    main()
