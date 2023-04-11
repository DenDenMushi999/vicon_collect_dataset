import os

import glob
from pathlib import Path
import argparse
import cv2 as cv



def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def resize_images(in_folder, out_folder, size):
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    fdirs = glob.glob(str(in_folder/'*'))

    for fdir in fdirs:
        fname = Path(fdir).name
        print(fdir, os.path.isfile(fdir))
        
        create_dir(out_folder)
        img = cv.imread(fdir)
        if img is not None:
            cv.imwrite(str(out_folder/fname), cv.resize(img, size))
            print(f'Saved image {fdir}')
        else: 
            print(f'can\'t read image {fdir}')

if __name__ == '__main__' :
    in_folder = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/lab_cars/1_blurred_gan/image_2'
    out_folder = '/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/lab_cars/1_blurred_gan/image_2_1'    
    size = (640,480)
    resize_images(in_folder, out_folder, size)