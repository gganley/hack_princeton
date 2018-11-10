import hashlib
import urllib

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

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


def rgb2gray(rgb):
    """
    Taken from 411 project website

    Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    """

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grey / 255.


if __name__ == '__main__':
    testfile = urllib.request.urlretrieve
    files = ["Resources/face_res/facescrub_actors.txt", "Resources/face_res/facescrub_actresses.txt"]
    j = 0
    for f_name in files:
        for a in act:
            name = a.split()[1].lower()
            if not os.path.isdir("Resources/face_res/" + name):
                os.mkdir("Resources/face_res/" + name)
            i = 0
            for line in open(f_name):

                if a in line:
                    filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
                    # A version without timeout (uncomment in case you need to
                    # unsuppress exceptions, which timeout() does)
                    #
                    # timeout is used to stop downloading images which take too long to download
                    data = None
                    try:
                        data = testfile(line.split()[4], "Resources/face_res/" + name + "/" + filename)
                    except:
                        pass
                    if data is None:
                        continue
                    with open(data[0], 'rb') as f:
                        h = hashlib.sha256(f.read()).hexdigest()
                        f.close()
                        if line.split()[6] != h:
                            os.remove("Resources/face_res/" + name + "/" + filename)
                    if line.split()[6] != h:
                        continue
                    try:


                        img_arr = imread(data[0])
                        bounding_box = line.split()[5]
                        x1, y1, x2, y2 = list(map(int, (bounding_box.split(','))))
                        cropped = img_arr[y1:y2, x1:x2]
                        final = imresize(cropped, (227, 227))

                        imsave(
                            "Resources/face_res/color_cropped/" + name + str(i) + "." + line.split()[4].split('.')[-1],
                            final
                        )
                        j += 1
                        print("added " + str(j) + " images")
                        i += 1
                    except Exception as e:
                        print(e)
                        print(filename + " error!")
                        try:
                            os.remove("Resources/face_res/" + name + "/" + filename)
                        except:
                            pass