import argparse,logging,os
import mxnet as mx
import glob
from cfgs.config import cfg, read_cfg
import pprint
import numpy as np
import numpy as np
import symbol.cycleGAN as cycleGAN
from symbol.cycleGAN import ImagenetIter
from util.visualizer import *


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_generator(inputA, inputB, lamd):
    # calculate loss for inputA
    modG_A.forward(mx.io.DataBatch(data=inputA, label=None), is_train=True)
    fakeB = modG_A.get_outputs()
    modG_B.forward(mx.io.DataBatch(data=fakeB, label=None), is_train=True)
    cycleA = modG_B.get_outputs()
    
    # backward for cycle L1 loss for inputA and cycleA
    # l1lossA, grad = get_l1grad(cycleA[0], inputA[0])
    cycleLoss_excu.forward(cycle = cycleA[0], data = inputA[0], is_train=True)
    cycleLoss_excu.backward()
    cyclossA = cycleLoss_excu.outputs[0].asnumpy()[0]
    grad = lamd * cycleLoss_excu.grad_dict['cycle']
    modG_B.backward([grad])

    # backward for GAN loss
    label[:] = 1
    modD_B.forward(mx.io.DataBatch(data=fakeB, label=[label]), is_train=True)
    modD_B.backward()
    
    # loss of discriminator
    DlossB = modD_B.get_outputs()[0].asnumpy()[0]

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
    # l1lossB, grad = get_l1grad(cycleB[0], inputB[0])
    cycleLoss_excu.forward(cycle = cycleB[0], data = inputB[0], is_train=True)
    cycleLoss_excu.backward()
    cyclossB = cycleLoss_excu.outputs[0].asnumpy()[0]
    grad = lamd * cycleLoss_excu.grad_dict['cycle']
    modG_A.backward([grad])

    # backward for GAN loss
    label[:] = 1
    modD_A.forward(mx.io.DataBatch(data=fakeA, label=[label]), is_train=True)
    modD_A.backward()
    
    # loss of Discriminator
    DlossA = modD_A.get_outputs()[0].asnumpy()[0]

    modG_B.backward([modG_A.get_input_grads()[0] + modD_A.get_input_grads()[0]])

    # update Generator A and Generator B
    for gradsr, gradsf in zip(modG_A._exec_group.grad_arrays, gradG_A):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr += gradf
   

    for gradsr, gradsf in zip(modG_B._exec_group.grad_arrays, gradG_B):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr += gradf   
    

    gradG_A = [[grad.copyto(grad.context) for grad in grads] for grads in modG_A._exec_group.grad_arrays]
    gradG_B = [[grad.copyto(grad.context) for grad in grads] for grads in modG_B._exec_group.grad_arrays]

    # identity loss
    if True:
        lamb_iden = 1
        modG_A.forward(mx.io.DataBatch(data=inputB, label=None), is_train=True)
        indenB = modG_A.get_outputs()
        cycleLoss_excu.forward(cycle=indenB[0], data=inputB[0], is_train=True)
        cycleLoss_excu.backward()
        grad = lamb_iden * cycleLoss_excu.grad_dict['cycle']
        modG_A.backward([grad])

        modG_B.forward(mx.io.DataBatch(data=inputA, label=None), is_train=True)
        indenA = modG_B.get_outputs()
        cycleLoss_excu.forward(cycle=indenA[0], data=inputA[0], is_train=True)
        cycleLoss_excu.backward()
        grad = lamb_iden * cycleLoss_excu.grad_dict['cycle']
        modG_B.backward([grad])

        # update Generator A and Generator B
        for gradsr, gradsf in zip(modG_A._exec_group.grad_arrays, gradG_A):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf

        for gradsr, gradsf in zip(modG_B._exec_group.grad_arrays, gradG_B):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf

        gradG_A = [[grad.copyto(grad.context) for grad in grads] for grads in modG_A._exec_group.grad_arrays]
        gradG_B = [[grad.copyto(grad.context) for grad in grads] for grads in modG_B._exec_group.grad_arrays]

    return cyclossA, cyclossB, gradG_A, gradG_B, DlossA, DlossB


def train_discriminator(modD, real, fake):
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
           
    modD.update()
    return loss/2


def update_module(mod, grad):
    for gradsr, gradsf in zip(mod._exec_group.grad_arrays, grad):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr = gradf     
    mod.update()


def update_learningrate(lr, steps, mod):
    lrd = lr / steps
    mod._optimizer.lr = mod._optimizer.lr - lrd


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
    devs = mx.cpu() if cfg.gpus is None else [mx.gpu(int(i)) for i in cfg.gpus.split(',')]
    check_point = False
    load_model = False
    mode_path = './SavedModel'
    
    global modG_A, modG_B, modD_A, modD_B, cycleLoss_excu, label
    label = mx.nd.zeros((cfg.batch_size, 1, cfg.dataset.dh, cfg.dataset.dw), ctx=ctx)

    symG_A, symG_B, symD_A, symD_B = cycleGAN.get_symbol(cfg)
    # Generator A
    modG_A = mx.mod.Module(symbol=symG_A, data_names=(
        'dataA',), label_names=None, context=devs)
    modG_A.bind(data_shapes=[('dataA', (cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))],
               inputs_need_grad=True)
    modG_A.init_params(initializer=mx.init.Normal(0.02))
    modG_A.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })

    # Generator B
    modG_B = mx.mod.Module(symbol=symG_B, data_names=(
        'dataB',), label_names=None, context=devs)
    modG_B.bind(data_shapes=[('dataB', (cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))],
               inputs_need_grad=True)
    modG_B.init_params(initializer=mx.init.Normal(0.02))
    modG_B.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })


    # Discriminator A
    modD_A = mx.mod.Module(symbol=symD_A, data_names=(
        'dataC',), label_names=('labelC',), context=devs)
    modD_A.bind(data_shapes=[('dataC', (cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))], 
                label_shapes=[('labelC', (cfg.batch_size, 1, cfg.dataset.dh, cfg.dataset.dw))],
                inputs_need_grad=True)
    modD_A.init_params(initializer=mx.init.Normal(0.02))
    modD_A.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })

    # Discriminator B
    modD_B = mx.mod.Module(symbol=symD_B, data_names=(
        'dataD',), label_names=('labelD',), context=devs)
    modD_B.bind(data_shapes=[('dataD', (cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))], 
                label_shapes=[('labelD', (cfg.batch_size, 1, cfg.dataset.dh, cfg.dataset.dw))],
                inputs_need_grad=True)
    modD_B.init_params(initializer=mx.init.Normal(0.02))
    modD_B.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': wd,
            'beta1': beta1,
        })

    cycleLoss = cycleGAN.getAbsLoss()
    cycleLoss_excu = cycleLoss.simple_bind(ctx=ctx, grad_rep='write', 
                                           cycle=(cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w), 
                                           data=(cfg.batch_size, cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))
    
    # load params
    if load_model:
        modG_A.load_params(os.path.join(mode_path, 'generatorA'))
        modG_B.load_params(os.path.join(mode_path, 'generatorB'))
        modD_A.load_params(os.path.join(mode_path, 'discriminatorA'))
        modD_B.load_params(os.path.join(mode_path, 'discriminatorB'))

    # load train data to iterator
    dataA = glob.glob(os.path.join(cfg.dataset.path, 'trainA/*.jpg'))
    dataB = glob.glob(os.path.join(cfg.dataset.path, 'trainB/*.jpg'))
    dataA_iter = ImagenetIter(dataA, cfg.batch_size, (cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))
    dataB_iter = ImagenetIter(dataB, cfg.batch_size, (cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))

    # load test data to iterator
    testA = glob.glob(os.path.join(cfg.dataset.path, 'testA/*.jpg'))
    testB = glob.glob(os.path.join(cfg.dataset.path, 'testB/*.jpg'))
    testA_iter = ImagenetIter(testA, cfg.batch_size, (cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))
    testB_iter = ImagenetIter(testB, cfg.batch_size, (cfg.dataset.c, cfg.dataset.h, cfg.dataset.w))

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)

    test = 0
    for epoch in range(cfg.num_epoch):
        dataA_iter.reset()
        dataB_iter.reset()
        for npic in range(cfg.dataset.num_pics):
            inputA = dataA_iter.getdata()
            inputB = dataB_iter.getdata()
            l1lossA, l1lossB, gradG_A, gradG_B, DlossA, DlossB = train_generator(inputA, inputB, 10)
            modG_A.forward(mx.io.DataBatch(data=inputA, label=None), is_train=True)
            fakeB = modG_A.get_outputs()
            modG_B.forward(mx.io.DataBatch(data=inputB, label=None), is_train=True)
            fakeA = modG_B.get_outputs()
            lossD_A = train_discriminator(modD_A, inputA, fakeA)
            lossD_B = train_discriminator(modD_B, inputB, fakeB)

            # update modG and modD
            update_module(modG_A, gradG_A)
            update_module(modG_B, gradG_B)
            
            if npic % cfg.frequent == 0:
                print('epoch:', str(npic), str(npic), 'lossD_A:', lossD_A, 'lossD_B:', lossD_B, 'l1loss_a:',
                      l1lossA,'l1loss_b:', l1lossB, 'DlossA:', DlossA, 'DlossB:', DlossB)
        
         
        # apply model to test data and save result pics
        if test == cfg.dataset.num_pics / 3:
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
        visual(os.path.join(cfg.out_path, 'A_B_A' + str(epoch) + '.jpg'), A_B_A)
        visual(os.path.join(cfg.out_path, 'B_A_B' + str(epoch) + '.jpg'), B_A_B)

        ## save model
        modG_A.save_params(os.path.join(mode_path, 'generatorA'))
        modG_B.save_params(os.path.join(mode_path, 'generatorB'))
        modD_A.save_params(os.path.join(mode_path, 'discriminatorA'))
        modD_B.save_params(os.path.join(mode_path, 'discriminatorB'))
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='cycleGAN')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model_path', help='the loc to save model checkpoints', default='', type=str)

    args = parser.parse_args()
    logging.info(args)
    main()