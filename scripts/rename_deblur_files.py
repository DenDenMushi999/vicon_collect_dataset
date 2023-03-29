import glob
import shutil

indir = r'/home/dendenmushi/ros1_ws/src/DSP-SLAM/data/lab_cars/1_deblurred_25/image_0/'
# filename as 000123_fake_B.png

fnames = glob.glob(indir + '*.png')
print(fnames[:10])
for fname in fnames:
    shutil.move(fname, fname[:-11] + fname[-4:])
