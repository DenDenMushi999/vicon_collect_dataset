import os
import glob
import shutil

import numpy as np

indir = r'/media/dendav/Eurobot/datasets/realsense_d435i/12/image_2/'
time_stamp_dir = '/media/dendav/Eurobot/datasets/realsense_d435i/12/times.txt'
fps = 30
# filename as 000123_fake_B.png

fnames = glob.glob(indir + '*')

print(fnames[:10])

time_stamps = np.arange(0,len(fnames))*1/fps
print(time_stamps[:10])
with open( time_stamp_dir, 'w') as f:
    f.write('\n'.join(str(t) for t in time_stamps))

