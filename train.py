import mxnet as mx
import util.visualizer as visualizer
import data.data_iter as data_iter
from options.base_options import BaseOptions
from models.wgan import WGAN
from models.dcgan import DCGAN
import os

opt = BaseOptions().parse()

if opt.model == 'dcgan':
    model = DCGAN(opt)
elif opt.model == 'wgan':
    model = WGAN(opt)
elif opt.model == 'infogan':
    model = infoGAN(opt)
elif opt.model == 'cyclegan':
    model = cycleGAN(opt)
else:
    raise ValueError("Model [%s] not recognized." % opt.model)

# load data set
if opt.dataset == 'mnist':
    X_train, X_test = data_iter.get_mnist()
    train_iter = mx.io.NDArrayIter(X_train, batch_size=opt.batchSize)
elif opt.dataset == 'imagenet':
    train_iter = data_iter.ImagenetIter(opt.dataroot, opt.batchSize, (opt.input_nc, opt.fineSize, opt.fineSize))

visual = visualizer.visual
rand_iter = data_iter.RandIter(opt.batchSize, opt.Z)
model.fit(rand_iter, train_iter, visual)
