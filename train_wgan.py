import argparse,logging,os
import mxnet as mx
import glob
from cfgs.config import cfg, read_cfg
import pprint
import numpy as np
import numpy as np
import symbol.dcgan as DCGAN
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


    symG, symD = DCGAN.get_symbol(cfg)

    if cfg.dataset.data_type == 'mnist':
        X_train, X_test = get_mnist()
        train_iter = mx.io.NDArrayIter(X_train, batch_size=cfg.batch_size)
    else:
        train_iter = ImagenetIter(cfg.dataset.path, cfg.batch_size, (cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))
    rand_iter = RandIter(cfg.batch_size, n_rand)
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
    
    randz = mx.random.normal(0, 1.0, shape=(cfg.batch_size, cfg.dataset.n_rand, 1, 1))
    fix_noise = mx.io.DataBatch(data=[mx.ndarray.array(randz)], label=[])

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)

    for epoch in range(cfg.num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            # clip weights
            for params in modD._exec_group.param_arrays:
                for param in params:
                    mx.nd.clip(param, -cfg.network.wclip, cfg.network.wclip, out=param)
            
            rbatch = rand_iter.next()
            # generate fake data
            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            modD.forward(mx.io.DataBatch(outG, label=None), is_train=True)
            modD.backward([-mx.nd.ones((cfg.batch_size, 1))])
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            # update discriminator on real
            modD.forward(batch, is_train=True)
            modD.backward([mx.nd.ones((cfg.batch_size, 1))])
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            # update generator
            modD.forward(mx.io.DataBatch(outG, label=None), is_train=True)
            modD.backward([mx.nd.ones((cfg.batch_size, 1))])
            diffD = modD.get_input_grads()
            modG.backward(diffD)
            modG.update()

            if t % cfg.frequent == 0:
                modG.forward(fix_noise, is_train=True)
                outG = modG.get_outputs()
                visual(cfg.out_path+'GAN_%d_%d.jpg'%(epoch+1, t+1), outG[0].asnumpy())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='cycleGAN')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model_path', help='the loc to save model checkpoints', default='', type=str)

    args = parser.parse_args()
    logging.info(args)
    main()