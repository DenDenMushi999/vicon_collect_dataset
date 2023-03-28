import glob
import shutil
from pathlib import Path

indir = r'/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/lab_cars/8/image_0/'
# filename as 000123_fake_B.png

def sort_key(fname):
    return int(fname[-10:-4])

fdirs = sorted(glob.glob(indir + '*.png'), key=sort_key)
print(fdirs[:10])

for fdir in fdirs:
    fdir_p = Path(fdir)
    fname = fdir_p.name
    print(str(int(fname[:-4])-580).zfill(6)+'.png', fname) 
    shutil.move(fdir, str(fdir_p.parent/(str(int(fname[:-4])-580).zfill(6)+'.png')))
