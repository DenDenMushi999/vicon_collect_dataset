import os
import glob
import shutil

indir = r'/home/dendenmushi/projects/DeblurGAN/output/images/'
# filename as 000123_fake_B.png

real_img_regexp = '\d+_real_A*'
rem_fnames = glob.glob(indir + '*real_A.png')
fnames = glob.glob(indir + '*fake_B.png')

print(rem_fnames[:10])
for fname in rem_fnames:
    os.remove(fname)

print(fnames[:10])
for fname in fnames:
    print(fname[:-11] + fname[-4:])
    shutil.move(fname, fname[:-11] + fname[-4:])
