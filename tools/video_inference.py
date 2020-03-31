import argparse
import os
import shutil
import cv2 as cv
import numpy as np
from os import sys
from os import path
from pprint import pprint
from sys import exit
from tqdm import tqdm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import mmcv
from mmdet.apis import init_detector, inference_detector

FONT_SIZE = 18
PERSON_CLASS_INDEX = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outdir")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--codec", default="XVID")
    parser.add_argument("--outfile_extension", default=".avi")
    args = parser.parse_args()

    # prepare model
    model = buildModel(args.config, args.checkpoint)

    # prepare in/out video files
    assert path.isfile(args.infile)
    cap = cv.VideoCapture(args.infile)
    assert cap.isOpened()
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    os.makedirs(args.outdir, exist_ok=True)
    outfile = path.join(
        args.outdir, path.basename(path.splitext(args.infile)[0]) + "_"
        + args.codec.lower() + "_annotated" + args.outfile_extension)
    outcsv = path.join(outfile + ".csv")
    out = cv.VideoWriter(
        outfile, cv.VideoWriter_fourcc(*args.codec.upper()), fps, (w,h))

    # inference on video frames
    with open(outcsv,"w") as f:
        with tqdm(total=frameCount) as pbar:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    frameIndex = pbar.n
                    line = str(frameIndex)
                    boxes = runModel(model, frame)
                    for box in boxes:
                        x1,y1,x2,y2,score = (
                            round(box[0]), round(box[1]), round(box[2]),
                            round(box[3]), box[4])
                        line += ("\t"
                            + ("\t".join(list(map(str,[x1,y1,x2,y2,score])))))

                        if score >= args.score_threshold:
                            cv.rectangle(frame, (x1,y1), (x2,y2), (255,0,0))
                            img = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img)
                            font = ImageFont.truetype(
                                path.join(
                                    getScriptPath(), "DejaVuSansCondensed.ttf"),
                                FONT_SIZE)
                            draw.text(
                                (x1,y1-FONT_SIZE), "{:.1f}".format(score),
                                (0,0,255), font=font)
                            frame = np.array(img)

                    f.write(line+"\n")
                    out.write(frame)
                    pbar.update()
                else:
                    break

    # close in/out video files
    cap.release()
    out.release()

def runModel(model, frame):
    return inference_detector(model, frame)[PERSON_CLASS_INDEX]

def buildModel(config, checkpoint):
    return init_detector(config, checkpoint, device='cuda:0')

def getScriptPath():
    return path.dirname(path.realpath(sys.argv[0]))



if __name__ == '__main__':
    main()
