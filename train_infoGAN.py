import argparse,logging,os
import mxnet as mx
import glob
from cfgs.config import cfg, read_cfg
import pprint
import numpy as np
import numpy as np
import symbol.infoGAN as infoGAN
from symbol.cycleGAN import ImagenetIter
from util.visualizer import *
from data.data_iter import *


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class cusDataBatch(object):
    """docstring for cusDataBatch"""

    def __init__(self, data, c, label):
        self.data = data
        self.label = [label, c]
        self.pad = 0

def main():
    # start program
    read_cfg(args.cfg)
    if args.gpus:
        cfg.gpus = args.gpus
    if args.model_path:
        cfg.model_path = args.model_path
    pprint.pprint(cfg)
   
    lr = cfg.train.lr
    beta1 = cfg.train.beta1
    wd = cfg.train.wd
    ctx = mx.gpu(0)
    check_point = False
    n_rand = cfg.dataset.n_rand
    n_class = cfg.dataset.n_class

    symG, symD, l1loss, group = infoGAN.get_symbol(cfg)

    if cfg.dataset.data_type == 'mnist':
        X_train, X_test = get_mnist()
        train_iter = mx.io.NDArrayIter(X_train, batch_size=cfg.batch_size)
    else:
        train_iter = ImagenetIter(cfg.dataset.path, cfg.batch_size, (cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))
    rand_iter = RandIter(cfg.batch_size, n_rand+n_class)
    label = mx.nd.zeros((cfg.batch_size,), ctx=ctx)

    modG = mx.mod.Module(symbol=symG, data_names=(
        'rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })
    mods = [modG]

    modD = mx.mod.Module(symbol=symD, data_names=(
        'data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (cfg.batch_size,))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })
    mods.append(modD)

    modGroup = mx.mod.Module(symbol=group, data_names=(
        'data',), label_names=('label', 'c'), context=ctx)
    modGroup.bind(data_shapes=[('data', (cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))],
              label_shapes=[('label', (cfg.batch_size,)), ('c', (cfg.batch_size, cfg.dataset.n_class,))],
              inputs_need_grad=True
              )
    modGroup.init_params(initializer=mx.init.Normal(0.02))
    modGroup.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })
    mods.append(modGroup)

    randz = mx.random.normal(0, 1.0, shape=(cfg.batch_size, cfg.dataset.n_rand, 1, 1))
    ids = np.array([np.eye(n_class)[:8, :] for _ in range(8)]).reshape(cfg.batch_size, cfg.dataset.n_class, 1, 1)
    fix_noise = mx.io.DataBatch(data=[mx.ndarray.concat(randz, 
                                                        mx.ndarray.array(ids.reshape(cfg.batch_size, 
                                                        cfg.dataset.n_class, 1, 1)))], label=[])

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)

    for epoch in range(cfg.num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            # generate fake data
            rbatch = rand_iter.next()
            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            label[:] = 0
            c = mx.ndarray.array(rbatch.data[0].asnumpy()[:, n_rand:n_rand+n_class, :, :].reshape(cfg.batch_size, n_class))

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
            c = mx.ndarray.array(rbatch.data[0].asnumpy()[:, n_rand:n_rand+n_class, :, :].reshape(cfg.batch_size, n_class))
            cusData = cusDataBatch(data=outG, c=c, label=label)
            modGroup.forward(cusData, is_train=True)
            modGroup.backward()
            l1_loss = modGroup.get_outputs()[1].asnumpy()[0]

            diffD = modGroup.get_input_grads()
            modG.backward(diffD)
            modG.update()

            if t % cfg.frequent == 0:
                print('epoch:', epoch+1, 'iteration: ', t, 'l1 loss: ', l1_loss)

            if t % cfg.frequent == 0:
                modG.forward(fix_noise, is_train=True)
                outG = modG.get_outputs()
                visual(cfg.out_path+'info_%d_%d.jpg'%(epoch+1, t+1), outG[0].asnumpy())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='cycleGAN')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model_path', help='the loc to save model checkpoints', default='', type=str)

    args = parser.parse_args()
    logging.info(args)
    main()