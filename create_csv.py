#! /usr/bin/python3

import argparse, json, os, shutil
from os import sys,path
from pprint import pprint
from sys import exit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detections")
    parser.add_argument("annotations")
    args = parser.parse_args()

    imageIdToName = {}
    detectionsPerImage = {}
    with open(args.annotations) as f:
        a = json.load(f)
        for image in a["images"]:
            imageIdToName[image["id"]] = image["file_name"]
            detectionsPerImage[image["file_name"]] = []

    with open(args.detections) as f:
        d = json.load(f)
        for det in d:
            if det["category_id"] == 1:
                image = imageIdToName[det["image_id"]]
                bbox = [
                    det["bbox"][0],
                    det["bbox"][1],
                    det["bbox"][0]+det["bbox"][2],
                    det["bbox"][1]+det["bbox"][3]
                ]
                detectionsPerImage[image] += bbox + [det["score"]]

    with open(path.splitext(args.detections)[0]+".csv","w") as f:
        for image,detections in detectionsPerImage.items():
            line = [image]+detections
            line = list(map(str,line))
            line = "\t".join(line)
            f.write(line+"\n")



if __name__ == '__main__':
    main()
