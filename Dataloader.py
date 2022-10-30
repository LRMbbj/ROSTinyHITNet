import time
import torch
import cv2
import numpy as np


class StreamLoader:
    def __init__(self):
        print("Loader Created...")
        self.imgs = [np.array([[]]*6), np.array([[]]*6)]
        self.enable = True

    def update(self, limg, rimg):
        self.limg = limg
        self.rimg = rimg

    def updateSplit(self, img, index, tag):
        self.imgs[0 if tag == "left" else 1][index] = img

    def Stop(self):
        self.enable = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.imgs[0], self.imgs[1]

    def __len__(self):
        # 1E12 frames = 32 streams at 30 FPS for 30 years
        return (1, 1)
