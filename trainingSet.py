import imageio
import numpy as np
import torch
from pylab import *
from scipy.misc import imresize
from torch.autograd import Variable

import myalexnet as a_net

FILE_NAME = 'this_thing.html'
TRAINING_SET_OUTPUT_DIR = 'training_data'
IMAGE_SET_OUTPUT_DIR = "training_images"
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
                with open(TRAINING_SET_OUTPUT_DIR + "/tset" + str(cnt // 10) + ".html", 'w') as f2:
                    f2.write(header)
                    f2.writelines(l)
                    f2.write(tail)
                l = []

            if cnt > 3000:
                break


def processImage():
    for i in range(1, 301):
        img_arr = imageio.imread(TRAINING_SET_OUTPUT_DIR + "/tset" + str(i) + ".html.pdf.jpg")
        img_arr = img_arr[0:300, 0:300]
        img_arr = imresize(img_arr, 227 / 300, 'nearest')
        imageio.imwrite(
            IMAGE_SET_OUTPUT_DIR + "/tset" + str(i) + "_cropped.html.pdf.jpg",
            img_arr
        )


def createTrainingSet():
    x = []
    y = np.zeros((0, 2))

    with open("test2.txt") as f:
        for line in f:
            idx = int(line)
            onehot = np.zeros(2)
            onehot[idx] = 1
            y = np.vstack((y, onehot))

    for i in range(151, 301):
        im = np.array(imageio.imread(IMAGE_SET_OUTPUT_DIR + "/tset" + str(i) + "_cropped.html.pdf.jpg"))
        im = im.T
        im = np.array((im[0].T, im[1].T, im[2].T)) / 255.0

        x.append(im)

    return np.array(x), np.array(y)


if __name__ == '__main__':
    x, y = createTrainingSet()
    train_x, train_y = x[0:100], y[0:100]
    valid_x, valid_y = x[101:130], y[101:130]
    test_x, test_y = x[131:150], y[131:150]

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    alexnet = a_net.MyAlexNet()
    train_x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    test_x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

    test_x = alexnet.forward2(test_x)
    test_x = test_x.data.numpy()

    train_x = alexnet.forward2(train_x)
    train_x = train_x.data.numpy()

    valid_x = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
    valid_x = alexnet.forward2(valid_x)

    dim_x = train_x.shape[1]
    dim_h = 30
    dim_out = train_y.shape[1]

    train_idx = np.random.permutation(range(train_x.shape[0]))[:150]
    v_validation_loss_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)

    in_layer = torch.nn.Linear(dim_x, dim_h)
    in_layer.weight = torch.nn.Parameter(torch.zeros(dim_h, dim_x))
    mid_layer = torch.nn.Linear(dim_h, dim_out)
    mid_layer.weight = torch.nn.Parameter(torch.zeros(dim_out, dim_h))

    model = torch.nn.Sequential(
        in_layer,
        torch.nn.Tanh(),
        mid_layer,
        torch.nn.Softmax(dim=-1)
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    validation_loss_y, validation_loss_v = [], []
    batch_size = 32
    t = int(train_idx.shape[0] / batch_size)
    k = 0
    for j in range(t):
        x = Variable(torch.from_numpy(train_x[train_idx[j * batch_size: (j + 1) * batch_size]]),
                     requires_grad=False).type(dtype_float)
        t_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx[j * batch_size: (j + 1) * batch_size]], 1)),
                             requires_grad=False).type(dtype_long)
        for i in range(10000 // t):
            y_pred = model(x)
            loss = loss_fn(y_pred, t_classes)
            v_loss = loss_fn(model(valid_x), v_validation_loss_classes)

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

    for i in range(y_pred.shape[0]):
        if argmax(y_pred[i]) == argmax(test_y[i]):
            k += 1
    print(k / y_pred.shape[0])


