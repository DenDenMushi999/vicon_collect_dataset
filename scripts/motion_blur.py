import os

import glob
from pathlib import Path
import argparse

import cv2 as cv
import numpy as np

def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

in_dir = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/freiburg_cars/Car001_end/image_0'
out_dir = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/freiburg_cars/Car001_blured_25/image_0'

in_dir = Path(in_dir)
out_dir = Path(out_dir)
create_dir(out_dir)

fdirs = glob.glob(str(in_dir/'*'))

kernel_size = 25
kernel_h = np.zeros((kernel_size, kernel_size))
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
kernel_h /= kernel_size


for fdir in fdirs:
    fname = Path(fdir).name
    img = cv.imread(fdir)
    
    if img is not None:
        horizonal_mb = cv.filter2D(img, -1, kernel_h)  
        cv.imwrite( str(out_dir/fname), horizonal_mb)
        print(f'Saved image {fdir}')
    else: 
        print(f'can\'t read image {fdir}')
