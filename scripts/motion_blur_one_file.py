import os

import glob
from pathlib import Path
import argparse

import cv2 as cv
import numpy as np

def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

in_dir = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/freiburg_cars/Car001_blurred_25/image_0/000050.png'

kernel_size = 40
kernel_h = np.zeros((kernel_size, kernel_size))
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
kernel_h /= kernel_size

img = cv.imread(in_dir)

if img is not None:
    horizonal_mb = cv.filter2D(img, -1, kernel_h)  
    cv.imwrite( 'blurred.png', horizonal_mb)
    print(f'Saved image ')
else: 
    print(f'can\'t read image {in_dir}')
