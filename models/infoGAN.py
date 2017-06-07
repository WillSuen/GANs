from __future__ import print_function
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
import logging
import cv2
from datetime import datetime
import os

# make infoGAN structure
TINY = 1e-8


def make_dcgan_sym(ngf, ndf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(
        4, 4), num_filter=ngf * 8, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ngf * 4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ngf * 2, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ngf, no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ndf, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ndf * 2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ndf * 4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4, 4), stride=(
        2, 2), pad=(1, 1), num_filter=ndf * 8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(
        4, 4), num_filter=1, no_bias=no_bias)
    d5 = mx.sym.Flatten(d5)

    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')

    fc = mx.sym.Convolution(dact4, name='q', kernel=(
        4, 4), num_filter=128, no_bias=no_bias)
    fcn = BatchNorm(fc, name='fcn', fix_gamma=fix_gamma, eps=eps)
    fcat = mx.sym.LeakyReLU(fcn, name='fcat', act_type='leaky', slope=0.2)
    q = mx.sym.FullyConnected(data=fcat, num_hidden=10, name='FC1')

    # data
    c = mx.sym.Variable('c')
    # propc = mx.sym.Variable('prob')
    
    # l1loss_ = mx.sym.softmax_cross_entropy(q, latentC, name='SM1')
    
    propx = mx.sym.SoftmaxActivation(data=q)
    l1loss_ = -200*mx.sym.mean(mx.sym.sum(mx.sym.log(propx + TINY) * c, axis=1)) # + mx.sym.mean(mx.sym.sum(mx.sym.log(propc + TINY) * latentC, axis=1))

    l1loss = mx.sym.MakeLoss(name='l1loss', data=l1loss_)
    group = mx.symbol.Group([dloss, l1loss])

    return gout, dloss, l1loss, group

## Get mnist data set
def get_mnist():
    import new_fetch_mnist
    new_fetch_mnist.fetch_mnist()
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

## Random number iterator
class RandIter(mx.io.DataIter):

    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim + 10, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        randz = mx.random.normal(0, 1.0, shape=(
            self.batch_size, self.ndim, 1, 1))
        ids = np.random.multinomial(1, [1 / 10.0] * 10, size=(64))
        return [mx.ndarray.concat(randz, mx.ndarray.array(ids.reshape(self.batch_size, 10, 1, 1)))]
    
## Image Iterator
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
    
def fill_buf(buf, i, img, shape):
    n = buf.shape[0] / shape[1]
    m = buf.shape[1] / shape[0]

    sx = (i % m) * shape[0]
    sy = (i / m) * shape[1]
    buf[sy:sy + shape[1], sx:sx + shape[0], :] = img
    
def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(dirname, title), buff)
    cv2.waitKey(1)
    
    
logging.basicConfig(level=logging.DEBUG)

# =============setting============
dataset = 'mnist'
imgnet_path = './train.rec'
ndf = 64
ngf = 64
nc = 3
batch_size = 64
Z = 100
lr = 0.0002
beta1 = 0.5
ctx = mx.gpu(0)
check_point = False

symG, symD, l1loss, group = make_dcgan_sym(ngf, ndf, nc)

if dataset == 'mnist':
    X_train, X_test = get_mnist()
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size)
elif dataset == 'imagenet':
    train_iter = ImagenetIter(imgnet_path, batch_size, (3, 64, 64))
rand_iter = RandIter(batch_size, Z)
label = mx.nd.zeros((batch_size,), ctx=ctx)

modG = mx.mod.Module(symbol=symG, data_names=(
    'rand',), label_names=None, context=ctx)
modG.bind(data_shapes=rand_iter.provide_data)
modG.init_params(initializer=mx.init.Normal(0.02))
modG.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })
mods = [modG]

modD = mx.mod.Module(symbol=symD, data_names=(
    'data',), label_names=('label',), context=ctx)
modD.bind(data_shapes=train_iter.provide_data,
          label_shapes=[('label', (batch_size,))],
          inputs_need_grad=True)
modD.init_params(initializer=mx.init.Normal(0.02))
modD.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })
mods.append(modD)

modGroup = mx.mod.Module(symbol=group, data_names=(
    'data',), label_names=('label', 'c'), context=ctx)
modGroup.bind(data_shapes=[('data', (64, 3, 64, 64))],
          label_shapes=[('label', (batch_size,)), ('c', (64, 10,))],
          inputs_need_grad=True
          )
modGroup.init_params(initializer=mx.init.Normal(0.02))
modGroup.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })
mods.append(modGroup)

class cusDataBatch(object):
    """docstring for cusDataBatch"""

    def __init__(self, data, c, label):
        self.data = data
        self.label = [label, c]
        self.pad = 0
        
randz = mx.random.normal(0, 1.0, shape=(
    64, 100, 1, 1))
ids = np.array([np.eye(10)[:8, :] for _ in range(8)]).reshape(64, 10, 1, 1)
fix_noise = mx.io.DataBatch(data=[mx.ndarray.concat(randz, mx.ndarray.array(ids.reshape(64, 10, 1, 1)))], label=[])

dirname = 'info_output'
if not os.path.exists(dirname):
    os.makedirs(dirname)

for epoch in range(100):
    train_iter.reset()
    for t, batch in enumerate(train_iter):
        # generate fake data
        rbatch = rand_iter.next()
        modG.forward(rbatch, is_train=True)
        outG = modG.get_outputs()

        # update discriminator on fake
        label[:] = 0
        c = mx.ndarray.array(rbatch.data[0].asnumpy()[:, 100:110, :, :].reshape(64, 10))
        # c = mx.ndarray.array(np.zeros((64, 10)))
        
        cusData = cusDataBatch(data=outG, c=c, label=label)
        modGroup.forward(cusData)
        modGroup.backward()
    
        gradD = [[grad.copyto(grad.context) for grad in grads]
                    for grads in modGroup._exec_group.grad_arrays]

        # update discriminator on real
        label[:] = 1
        c = mx.ndarray.array(np.zeros((64, 10)))
        cusData = cusDataBatch(data=batch.data, c=c, label=label)
        modGroup.forward(cusData, is_train=True)
        modGroup.backward()
        
        # update discriminator
        for gradsr, gradsf in zip(modGroup._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        modGroup.update()

        # update generator
        label[:] = 1
        c = mx.ndarray.array(rbatch.data[0].asnumpy()[:, 100:110, :, :].reshape(64, 10))
        cusData = cusDataBatch(data=outG, c=c, label=label)
        modGroup.forward(cusData, is_train=True)
        modGroup.backward()
        l1_loss = modGroup.get_outputs()[1].asnumpy()
        
        diffD = modGroup.get_input_grads()
        modG.backward(diffD)
        modG.update()
        
        if t % 10 == 0:
            print('epoch:', epoch+1, 'iteration: ', t, 'l1 loss: ', l1_loss)
        
        if t % 100 == 0:
            modG.forward(fix_noise, is_train=True)
            outG = modG.get_outputs()
            visual('info_' + str(epoch+1) + '_' + str(t+1) + '.jpg', outG[0].asnumpy())