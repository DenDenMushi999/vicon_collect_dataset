import glob
import shutil

fnames = glob.glob('*.jpg')
for fname in fnames:
    shutil.move(fname, fname[:-4]+'.png')
