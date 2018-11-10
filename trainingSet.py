import hashlib
import urllib

import imageio
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import *
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters


FILE_NAME = 'this_thing.html'
TRAINING_SET_OUTPUT_DIR = 'training_data'

header = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
'''

tail = '''
</body>
</html>
'''


def preProcessRows():
    with open(FILE_NAME, 'r') as f:
        l = []
        for cnt, line in enumerate(f):
            l.append(line)
            if cnt % 10 == 0 and cnt != 0:
                with open(TRAINING_SET_OUTPUT_DIR + "/tset"+str(cnt//10)+".html", 'w') as f2:
                    f2.write(header)
                    f2.writelines(l)
                    f2.write(tail)
                l = []

            if cnt > 3000:
                break


def processImage(img):
    img_arr = imageio.imread("tset12.html.pdf.jpg")
    img_arr = img_arr[0:300, 0:300]
    img_arr = imresize(img_arr, dsize=(227, 227))
    imageio.imwrite(
        "test.jpg",
        img_arr
    )



if __name__ == '__main__':
    processImage(None)