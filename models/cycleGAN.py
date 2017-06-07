import sys
sys.path.append('/data/guest_users/weisun/src')
sys.path.append('../')
import mxnet as mx
import glob
import os
from util.visualizer import *


def make_symG(data, ngf=64, use_dropout=False, n_blocks=9, no_bias=False, fix_gamma=False, eps=1e-5 + 1e-12):
    
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
        reslayer = mx.sym.Convolution(reslayer_out, name='resc1_%d' % i, kernel=(3, 3), pad=(1, 1), num_filter=ngf*4, no_bias=no_bias)
        reslayer = BatchNorm(reslayer, name='resbn1_%d' % i, fix_gamma=fix_gamma, eps=eps)
        reslayer = mx.sym.Activation(reslayer, name='resact%d' % i, act_type='relu')
        reslayer = mx.sym.Convolution(reslayer, name='resc2_%d' % i, kernel=(3, 3), pad=(1, 1), num_filter=ngf*4, no_bias=no_bias)
        reslayer = BatchNorm(reslayer, name='resbn2_%d' % i, fix_gamma=fix_gamma, eps=eps)
        reslayer_out = reslayer_out + reslayer
    
    # Deconvolution Layer
    d1 = mx.sym.Deconvolution(reslayer_out, name='d1', kernel=(3, 3), pad=(1, 1), stride=(2, 2), adj=(1, 1), num_filter=ngf*2, no_bias=no_bias)
    dbn1 = BatchNorm(d1, name='dbn1', fix_gamma=fix_gamma, eps=eps)
    dact1 = mx.sym.Activation(dbn1, name='dact1', act_type='relu')
    
    d2 = mx.sym.Deconvolution(dact1, name='d2', kernel=(3, 3), pad=(1, 1), stride=(2, 2), adj=(1, 1), num_filter=ngf, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.Activation(dbn2, name='dact2', act_type='relu')
    
    c4 = mx.sym.Convolution(dact2, name='c4', kernel=(7, 7), pad=(3, 3), stride=(1, 1), num_filter=3, no_bias=no_bias)
    gout = mx.sym.Activation(c4, name='gout', act_type='tanh')
    
    return gout
    
    
def make_symD(data, label, ndf=64, no_bias=False, fix_gamma=False, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    d1 = mx.sym.Convolution(data, name='d1', kernel=(4, 4), stride=(
        2, 2), pad=(2, 2), num_filter=ndf, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4, 4), stride=(
        2, 2), pad=(2, 2), num_filter=ndf * 2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4, 4), stride=(
        2, 2), pad=(2, 2), num_filter=ndf * 4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4, 4), stride=(
        1, 1), pad=(2, 2), num_filter=ndf * 8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(
        4, 4), pad=(2, 2), num_filter=1, no_bias=no_bias)

    mseloss_ = mx.sym.mean(mx.sym.square(d5 - label))
    mseloss = mx.sym.MakeLoss(data=mseloss_, name='mean_square_loss')
    
    return mseloss


def make_cycleGAN(ngf=64, ndf=64):
    # Generator
    dataA = mx.sym.Variable('dataA')
    dataB = mx.sym.Variable('dataB')
    symG_A = make_symG(dataA, ngf=64)
    symG_B = make_symG(dataB, ngf=64)
    
    # Discriminator
    dataC = mx.sym.Variable('dataC')
    dataD = mx.sym.Variable('dataD')
    labelC = mx.sym.Variable('labelC')
    labelD = mx.sym.Variable('labelD')
    symD_A = make_symD(dataC, labelC, ndf=64)
    symD_B = make_symD(dataD, labelD, ndf=64)
    
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


def get_l1grad(cycle, real, lamb=10, sigma=1000):
    l1_g = (cycle.asnumpy() - real.asnumpy())
    l1_loss = np.mean(np.abs(l1_g))
    l1_g = l1_g * sigma * sigma
    l1_g[l1_g > 1] = 1
    l1_g[l1_g < -1] = -1
    grad = mx.ndarray.array(l1_g * lamb *(1.0/(256.0*256.0*3.0)), ctx=ctx)
    return l1_loss, grad


def update_generator(inputA, inputB):
    # calculate loss for inputA
    modG_A.forward(mx.io.DataBatch(data=inputA, label=None), is_train=True)
    fakeB = modG_A.get_outputs()
    modG_B.forward(mx.io.DataBatch(data=fakeB, label=None), is_train=True)
    cycleA = modG_B.get_outputs()

    # backward for cycle L1 loss for inputA and cycleA
    l1lossA, grad = get_l1grad(cycleA[0], inputA[0])
    modG_B.backward([grad])

    # backward for GAN loss
    label[:] = 1
    modD_B.forward(mx.io.DataBatch(data=fakeB, label=[label]), is_train=True)
    modD_B.backward()

    modG_A.backward([modG_B.get_input_grads()[0] + modD_B.get_input_grads()[0]])

    # save gradients for future update
    gradG_A = [[grad.copyto(grad.context) for grad in grads] for grads in modG_A._exec_group.grad_arrays]
    gradG_B = [[grad.copyto(grad.context) for grad in grads] for grads in modG_B._exec_group.grad_arrays]

    # Calculate loss for inputB
    modG_B.forward(mx.io.DataBatch(data=inputB, label=None), is_train=True)
    fakeA = modG_B.get_outputs()
    modG_A.forward(mx.io.DataBatch(data=fakeA, label=None), is_train=True)
    cycleB = modG_A.get_outputs()

    # backward for cycle L1 loss for inputB and cycleB
    l1lossB, grad = get_l1grad(cycleB[0], inputB[0])
    modG_A.backward([grad])

    # backward for GAN loss
    label[:] = 1
    modD_A.forward(mx.io.DataBatch(data=fakeA, label=[label]), is_train=True)
    modD_A.backward()

    modG_B.backward([modG_A.get_input_grads()[0] + modD_A.get_input_grads()[0]])

    # update Generator A and Generator B
    for gradsr, gradsf in zip(modG_A._exec_group.grad_arrays, gradG_A):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr += gradf
    modG_A.update()

    for gradsr, gradsf in zip(modG_B._exec_group.grad_arrays, gradG_B):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr += gradf
    modG_B.update()
    
    return l1lossA, l1lossB


def update_discriminator(modD, real, fake):
    # train with real data
    label[:] = 1
    modD.forward(mx.io.DataBatch(data=real, label=[label]), is_train=True)
    modD.backward()
    # save gradient for future use
    gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]
    
    # loss of discriminator
    loss = modD.get_outputs()[0].asnumpy()[0]
    
    # train with fake data
    label[:] = 0
    modD.forward(mx.io.DataBatch(data=fake, label=[label]), is_train=True)
    modD.backward()
    loss += modD.get_outputs()[0].asnumpy()[0]
    for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr += gradf
            gradr *= 0.5
    modD.update()
    return loss
    
# start program   
ndf = 64
ngf = 64
nc = 3
batch_size = 1
lr = 0.0002
beta1 = 0.5
ctx = mx.gpu(0)
check_point = False
load_model = False
mode_path = './SavedModel'
label = mx.nd.zeros((batch_size, 1, 35, 35), ctx=ctx)


symG_A, symG_B, symD_A, symD_B = make_cycleGAN(ngf, ndf)
# Generator A
modG_A = mx.mod.Module(symbol=symG_A, data_names=(
    'dataA',), label_names=None, context=ctx)
modG_A.bind(data_shapes=[('dataA', (batch_size, 3, 256, 256))],
           inputs_need_grad=True)
modG_A.init_params(initializer=mx.init.Normal(0.02))
modG_A.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })

# Generator B
modG_B = mx.mod.Module(symbol=symG_B, data_names=(
    'dataB',), label_names=None, context=ctx)
modG_B.bind(data_shapes=[('dataB', (batch_size, 3, 256, 256))],
           inputs_need_grad=True)
modG_B.init_params(initializer=mx.init.Normal(0.02))
modG_B.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })


# Discriminator A
modD_A = mx.mod.Module(symbol=symD_A, data_names=(
    'dataC',), label_names=('labelC',), context=ctx)
modD_A.bind(data_shapes=[('dataC', (batch_size, 3, 256, 256))], 
            label_shapes=[('labelC', (batch_size, 1, 35, 35))],
            inputs_need_grad=True)
modD_A.init_params(initializer=mx.init.Normal(0.02))
modD_A.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })

# Discriminator B
modD_B = mx.mod.Module(symbol=symD_B, data_names=(
    'dataD',), label_names=('labelD',), context=ctx)
modD_B.bind(data_shapes=[('dataD', (batch_size, 3, 256, 256))], 
            label_shapes=[('labelD', (batch_size, 1, 35, 35))],
            inputs_need_grad=True)
modD_B.init_params(initializer=mx.init.Normal(0.02))
modD_B.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })

# load params
if load_model:
    modG_A.load_params(os.path.join(mode_path, 'generatorA'))
    modG_B.load_params(os.path.join(mode_path, 'generatorB'))
    modD_A.load_params(os.path.join(mode_path, 'discriminatorA'))
    modD_B.load_params(os.path.join(mode_path, 'discriminatorB'))

# load train data to iterator
data_root = '/data/guest_users/weisun/datasets/maps/'
dataA = glob.glob(os.path.join(data_root, 'trainA/*.jpg'))
dataB = glob.glob(os.path.join(data_root, 'trainB/*.jpg'))
dataA_iter = ImagenetIter(dataA, batch_size, (3, 256, 256))
dataB_iter = ImagenetIter(dataB, batch_size, (3, 256, 256))

# load test data to iterator
testA = glob.glob(os.path.join(data_root, 'testA/*.jpg'))
testB = glob.glob(os.path.join(data_root, 'testB/*.jpg'))
testA_iter = ImagenetIter(testA, batch_size, (3, 256, 256))
testB_iter = ImagenetIter(testB, batch_size, (3, 256, 256))

dirname = './outputs/maps_new'
if not os.path.exists(dirname):
    os.makedirs(dirname)

test = 0
for i in range(100):
    if test == 200:
        testA_iter.reset()
        testB_iter.reset()
        test = 0
    A_B_As = []
    B_A_Bs = []
    for _ in range(3):
        testA = testA_iter.getdata()
        testB = testB_iter.getdata()
        test += 1
        # visualize A-B-A
        modG_A.forward(mx.io.DataBatch(data=testA, label=None), is_train=True)
        fakeB = modG_A.get_outputs()
        modG_B.forward(mx.io.DataBatch(data=fakeB, label=None), is_train=True)
        cycleA = modG_B.get_outputs()
        A_B_As.append(np.concatenate((testA[0].asnumpy(), fakeB[0].asnumpy(), cycleA[0].asnumpy())))

        # visualize B-A-B
        modG_B.forward(mx.io.DataBatch(data=testB, label=None), is_train=True)
        fakeA = modG_B.get_outputs()
        modG_A.forward(mx.io.DataBatch(data=fakeA, label=None), is_train=True)
        cycleB = modG_A.get_outputs()
        B_A_Bs.append(np.concatenate((testB[0].asnumpy(), fakeA[0].asnumpy(), cycleB[0].asnumpy())))

    A_B_A = np.concatenate((A_B_As[0], A_B_As[1], A_B_As[2]))
    B_A_B = np.concatenate((B_A_Bs[0], B_A_Bs[1], B_A_Bs[2]))
    visual(os.path.join(dirname, 'A_B_A' + str(i) + '.jpg'), A_B_A)
    visual(os.path.join(dirname, 'B_A_B' + str(i) + '.jpg'), B_A_B)

    dataA_iter.reset()
    dataB_iter.reset()
    for j in range(1000):
        inputA = dataA_iter.getdata()
        inputB = dataB_iter.getdata()
        l1lossA, l1lossB = update_generator(inputA, inputB)
        modG_A.forward(mx.io.DataBatch(data=inputA, label=None), is_train=True)
        fakeB = modG_A.get_outputs()
        modG_B.forward(mx.io.DataBatch(data=inputB, label=None), is_train=True)
        fakeA = modG_B.get_outputs()
        lossD_A = update_discriminator(modD_A, inputA, fakeA)
        lossD_B = update_discriminator(modD_B, inputB, fakeB)
        if j % 200 == 0:
            print 'epoch: ' + str(i) + '_' + str(j) + ' lossD_A: ' + str(lossD_A) + ' lossD_B: ' + str(lossD_B) + ' l1loss_a:' + str(l1lossA) + ' l1loss_b:' + str(l1lossB)
            # visual(os.path.join(dirname, 'fakeA'+str(i)+'_'+str(j/100)+'.jpg'), fakeA[0].asnumpy())
            # visual(os.path.join(dirname, 'fakeB'+str(i)+'_'+str(j/100)+'.jpg'), fakeB[0].asnumpy())
            # modG_A.forward(mx.io.DataBatch(data=testA, label=None), is_train=True)
            # modG_B.forward(mx.io.DataBatch(data=testB, label=None), is_train=True)
            # visual(os.path.join(dirname, 'testB'+str(i)+'_'+str(j/100)+'.jpg'), modG_A.get_outputs()[0].asnumpy())
            # visual(os.path.join(dirname, 'testA'+str(i)+'_'+str(j/100)+'.jpg'), modG_B.get_outputs()[0].asnumpy())

    if i % 10 == 0:
        modG_A.save_params(os.path.join(mode_path, 'generatorA'))
        modG_B.save_params(os.path.join(mode_path, 'generatorB'))
        modD_A.save_params(os.path.join(mode_path, 'discriminatorA'))
        modD_B.save_params(os.path.join(mode_path, 'discriminatorB'))
