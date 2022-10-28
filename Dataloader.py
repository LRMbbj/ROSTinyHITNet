import time
import torch
import cv2
import numpy as np


class LoadStreams:
    def __init__(self):
        print("Loader Created...")
        self.limg = []
        self.rimg = []
        self.enable = True

    def update(self, limg, rimg):
        self.limg = limg
        self.rimg = rimg

    def updateSplit(self, img, tag):
        if tag == 'l':
            self.limg = img
        if tag == 'r':
            self.rimg = img

    def Stop(self):
        self.enable = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.limg, self.rimg

    def __len__(self):
        # 1E12 frames = 32 streams at 30 FPS for 30 years
        return (1, 1)
