# GANs

This is mxnet implementation of Generative Adversarial Networks. Until now, I have done:  
### DCGAN (https://arxiv.org/pdf/1511.06434.pdf)

code get from MXNet examples

### InfoGAN (https://arxiv.org/pdf/1606.03657.pdf)

#### Train on mnist
`python train_infoGAN.py --cfg cfgs/infoGAN_mnist.yaml`

### Wassertein GAN (https://arxiv.org/pdf/1701.07875.pdf)

#### Train on mnist
`python train_wgan.py --cfg cfgs/wgan_mnist.yaml`

### Cycle GAN (https://arxiv.org/pdf/1703.10593.pdf)

#### Train on horse&zebra
`python train_cycleGAN.py --cfg cfgs/cycle_horse2zebra.yaml`
