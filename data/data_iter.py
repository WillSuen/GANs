import numpy as np
from sklearn.datasets import fetch_mldata
import logging
import cv2
from datetime import datetime
import mxnet as mx


def get_mnist():
    import fetch_mnist
    fetch_mnist.fetch_mnist()
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234)  # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64, 64)) for x in X])

    X = X.astype(np.float32) / (255.0 / 2) - 1.0
    X = X.reshape((70000, 1, 64, 64))
    X = np.tile(X, (1, 3, 1, 1))
    X_train = X[:60000]
    X_test = X[60000:]

    return X_train, X_test


class RandIter(mx.io.DataIter):

    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]


class ImagenetIter(mx.io.DataIter):

    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec=path,
            data_shape=data_shape,
            batch_size=batch_size,
            rand_crop=True,
            rand_mirror=True,
            max_crop_size=256,
            min_crop_size=192)
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data = data * (2.0 / 255.0)
        data -= 1
        return [data]
