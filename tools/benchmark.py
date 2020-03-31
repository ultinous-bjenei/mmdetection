import argparse
import json
import os
import shutil
import numpy as np
from os import sys
from os import path
from pprint import pprint
from sys import exit
from timeit import default_timer as timer
from pprint import pprint

import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.core import wrap_fp16_model

TEST_COUNT = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--width",  type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    model = model.cuda()
    data = torch.cuda.FloatTensor(
        args.batch_size, 3, args.height, args.width)#.fill_(0) # NCHW

    times = []
    for i in range(TEST_COUNT):
        start = timer()

        with torch.no_grad():
            model.forward_dummy(data)

        time = (timer() - start) * 1000
        print("{} {} forward time: {} ms".format(TEST_COUNT, i, time))
        times.append(time)

    print("\naverage forward time: {} ms".format(
        np.mean(times, dtype=np.float64)))



if __name__ == '__main__':
    main()
