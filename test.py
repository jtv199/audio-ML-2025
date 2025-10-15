from PIL import Image
import numpy as np
import os

paths = [os.path.join("work/trn_curated",f) for f in os.listdir("work/trn_curated") if f != ".DS_Store"]
paths.sort()
path = paths[4]
img1 = Image.open(f"{path}/mel.png")
img2 = Image.open(f"{path}/mel2.png")

np1 = np.array(img1)
np2 = np.array(img2)

for i in range(len(np1)):
    for j in range(len(np1[0,:,:])):
        if np1[i,j,0] != np2[i,j,0]:
            print(i,j, np1[i,j,0], np2[i,j,0])