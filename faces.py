from pylab import *
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import *
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters

from torch.autograd import Variable
import torch
import urllib.request
import hashlib

from scipy.io import loadmat

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

np.random.seed(648)
torch.manual_seed(648)

PATH_TO_IMAGES = "Resources/face_res/cropped"


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
    return grey / 255.0


def acc_vs_iterations(x, yt, yv):

    plt.xlabel("# iterations")
    plt.ylabel("loss")
    plt.axis([0, len(x), 1, 1.9])
    plt.plot(x, yt, 'r-', x, yv, 'b-')
    plt.show()


def construct_arrays(sets, set_name):
    x = np.zeros((0, 32 * 32))
    y = np.zeros((0, 6))
    for a in act:
        onehot = np.zeros(6)
        onehot[act.index(a)] = 1
        files = sets[a][set_name]
        for img in files:
            im = np.array(plt.imread("Resources/face_res/cropped/" + img).flatten())

            x = np.vstack((x, (im / 255.0)))
            y = np.vstack((y, onehot))

    return x, y


def visualize(w):
    img = np.reshape(w, (32, 32))
    plt.imshow(img, cmap="RdBu")
    plt.show()


def construct_sets():
    total_sets_for_actors = {}
    f = [f for f in os.listdir(PATH_TO_IMAGES) if (os.path.isfile(os.path.join("Resources/face_res/cropped", f)))]
    np.random.shuffle(f)
    print(len(f))

    for a in act:
        total_sets_for_actors[a] = {"cnt": 0, "training": [], "validation": [], "test": []}

    for file in f:
        for a in act:

            last_name = a.split(' ')[1].lower()

            if last_name in file:
                if total_sets_for_actors[a]["cnt"] < 10:
                    total_sets_for_actors[a]["validation"].append(file)

                elif total_sets_for_actors[a]["cnt"] < 30:
                    total_sets_for_actors[a]["test"].append(file)

                else:
                    total_sets_for_actors[a]["training"].append(file)
                total_sets_for_actors[a]["cnt"] += 1

    return total_sets_for_actors


def part8():
    images = construct_sets()

    train_x, train_y = construct_arrays(images, "training")
    test_x, test_y = construct_arrays(images, "test")
    valid_x, valid_y = construct_arrays(images, "validation")
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    dim_x = train_x.shape[1]
    dim_h = 30
    dim_out = train_y.shape[1]

    train_idx = np.random.permutation(range(train_x.shape[0]))[:600]

    v_x = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
    v_validation_loss_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)

    in_layer = torch.nn.Linear(dim_x, dim_h)
    in_layer.weight = torch.nn.Parameter(torch.zeros(dim_h, dim_x))
    out_layer = torch.nn.Linear(dim_h, dim_out)
    out_layer.weight = torch.nn.Parameter(torch.zeros(dim_out, dim_h))

    model = torch.nn.Sequential(
        in_layer,
        torch.nn.Tanh(),
        out_layer,
        torch.nn.Softmax()
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    validation_loss_y, validation_loss_v = [], []
    batch_size = 32
    t = int(train_idx.shape[0]/batch_size)
    k = 0
    for j in range(t):
        x = Variable(torch.from_numpy(train_x[train_idx[j * batch_size: (j + 1) * batch_size]]),
                     requires_grad=False).type(dtype_float)
        t_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx[j * batch_size: (j + 1) * batch_size]], 1)),
                             requires_grad=False).type(dtype_long)
        for i in range(100000//t):
            y_pred = model(x)
            loss = loss_fn(y_pred, t_classes)
            v_loss = loss_fn(model(v_x), v_validation_loss_classes)

            validation_loss_y.append(loss.data[0])
            validation_loss_v.append(v_loss.data[0])

            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()  # Compute the gradient
            optimizer.step()  # Use the gradient information to
            # make a step
            if i % 500 == 0:
                print("Batch: " + str(j + 1) + ", Iteration: " + str(i))

            k += 1

    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()

    performance = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
    print(performance)

    acc_vs_iterations(np.arange(0, k), validation_loss_y, validation_loss_v)
    for i in [0, 5]:
        i1 = np.argmax(out_layer.weight.data[i].numpy())
        unit1 = model[0].weight.data.numpy()[i1, :]
        visualize(unit1)


if __name__ == "__main__":
    part8()
