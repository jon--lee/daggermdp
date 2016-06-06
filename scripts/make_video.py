import cv2
import os
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')



def num_from_filename(filename):
    return int(filename.split('dagger', 1)[1][:-4])


in_path = 'images/'
out_path = "videos/dagger_video.mp4"
writer = None
writer = cv2.VideoWriter(out_path, fourcc, 5.0, (800, 600))

filenames = [ in_path + fn for fn in os.listdir(in_path) if fn.endswith('.png') ]
print filenames
filenames = sorted(filenames, key=num_from_filename)

for fn in filenames:
    im = cv2.imread(fn)
    cv2.imwrite('videos/' + fn, im)
    writer.write(im)
