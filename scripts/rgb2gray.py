import os

import glob
from pathlib import Path
import argparse
import cv2 as cv

in_dir = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/lab_cars/8/image_0'
out_dir = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/lab_cars/8/image_2'

in_dir = Path(in_dir)
out_dir = Path(out_dir)
fdirs = glob.glob(str(in_dir/'*'))


for fdir in fdirs:
    fname = Path(fdir).name
    print(fdir, os.path.isfile(fdir))
    img = cv.imread(fdir, cv.IMREAD_GRAYSCALE)
    if img is not None:
        cv.imwrite( str(out_dir/fname), img)
        print(f'Saved image {fdir}')
    else: 
        print(f'can\'t read image {fdir}')