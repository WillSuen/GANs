import mxnet as mx


def make_symG(data, cfg):
    ngf = cfg.network.ngf
    use_dropout = cfg.network.dropout
    n_blocks = cfg.network.n_blocks
    no_bias = False
    fix_gamma = False
    eps = 1e-5 + 1e-12
    
    BatchNorm = mx.sym.BatchNorm
    
    # Convolution BatchNorm RELU layer
    c1 = mx.sym.Convolution(data, name='c1', kernel=(7, 7), pad=(3, 3), stride=(1, 1), num_filter=ngf, no_bias=no_bias)
    cbn1 = BatchNorm(c1, name='cbn1', fix_gamma=fix_gamma, eps=eps)
    cact1 = mx.sym.Activation(cbn1, name='cact1', act_type='relu')
    
    c2 = mx.sym.Convolution(cact1, name='c2', kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=ngf*2, no_bias=no_bias)
    cbn2 = BatchNorm(c2, name='cbn2', fix_gamma=fix_gamma, eps=eps)
    cact2 = mx.sym.Activation(cbn2, name='cact2', act_type='relu')
    
    c3 = mx.sym.Convolution(cact2, name='c3', kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=ngf*4, no_bias=no_bias)
    cbn3 = BatchNorm(c3, name='cbn3', fix_gamma=fix_gamma, eps=eps)
    cact3 = mx.sym.Activation(cbn3, name='cact3', act_type='relu')
    
    # Resnet Block
    reslayer_out = cact3
    for i in range(n_blocks):
        reslayer = mx.sym.Convolution(reslayer_out, name='resc1_%d' % i, kernel=(3, 3), pad=(1, 1), 
                                      num_filter=ngf*4, no_bias=no_bias)
        reslayer = BatchNorm(reslayer, name='resbn1_%d' % i, fix_gamma=fix_gamma, eps=eps)
        reslayer = mx.sym.Activation(reslayer, name='resact%d' % i, act_type='relu')
        reslayer = mx.sym.Convolution(reslayer, name='resc2_%d' % i, kernel=(3, 3), pad=(1, 1), 
                                      num_filter=ngf*4, no_bias=no_bias)
        reslayer = BatchNorm(reslayer, name='resbn2_%d' % i, fix_gamma=fix_gamma, eps=eps)
        reslayer_out = reslayer_out + reslayer
    
    # Deconvolution Layer
    d1 = mx.sym.Deconvolution(reslayer_out, name='d1', kernel=(3, 3), pad=(1, 1), stride=(2, 2), 
                              adj=(1, 1), num_filter=ngf*2, no_bias=no_bias)
    dbn1 = BatchNorm(d1, name='dbn1', fix_gamma=fix_gamma, eps=eps)
    dact1 = mx.sym.Activation(dbn1, name='dact1', act_type='relu')
    
    d2 = mx.sym.Deconvolution(dact1, name='d2', kernel=(3, 3), pad=(1, 1), stride=(2, 2), 
                              adj=(1, 1), num_filter=ngf, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.Activation(dbn2, name='dact2', act_type='relu')
    
    c3 = mx.sym.Convolution(dact2, name='c4', kernel=(7, 7), pad=(3, 3), stride=(1, 1), num_filter=3, no_bias=no_bias)
    gout = mx.sym.Activation(c3, name='gout', act_type='tanh')
    
    return gout
    
    
def make_symD(data, label, cfg):
    ndf = cfg.network.ndf
    no_bias=False
    fix_gamma=False
    eps=1e-5 + 1e-12
    
    BatchNorm = mx.sym.BatchNorm
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
        1, 1), pad=(1, 1), num_filter=ndf * 8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(
        4, 4), pad=(1, 1), num_filter=1, no_bias=no_bias)

    mseloss_ = mx.sym.mean(mx.sym.square(d5 - label))
    mseloss = mx.sym.MakeLoss(data=mseloss_, name='mean_square_loss')
    
    return mseloss


def get_symbol(cfg):
    # Generator
    dataA = mx.sym.Variable('dataA')
    dataB = mx.sym.Variable('dataB')
    symG_A = make_symG(dataA, cfg)
    symG_B = make_symG(dataB, cfg)
    
    # Discriminator
    dataC = mx.sym.Variable('dataC')
    dataD = mx.sym.Variable('dataD')
    labelC = mx.sym.Variable('labelC')
    labelD = mx.sym.Variable('labelD')
    symD_A = make_symD(dataC, labelC, cfg)
    symD_B = make_symD(dataD, labelD, cfg)
    
    return symG_A, symG_B, symD_A, symD_B


# Image Iterator
class ImagenetIter(mx.io.DataIter):

    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.image.ImageIter(
            imglist=[[1, img] for img in path],
            data_shape=data_shape,
            batch_size=batch_size,
            path_root='./',
            )
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.next().data[0]
        data = data * (2.0 / 255.0)
        data -= 1
        return [data]


def getAbsLoss():
    data = mx.sym.Variable('data')
    cycle = mx.sym.Variable('cycle')
    diff = cycle - data
    abs_ = mx.sym.abs(diff)
    loss = mx.sym.mean(abs_)
    Absloss = mx.sym.MakeLoss(data=loss, name='AbsLoss')
    return Absloss
