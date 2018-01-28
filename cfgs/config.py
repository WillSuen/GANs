import numpy as np
from easydict import EasyDict as edict
import yaml

cfg = edict()
cfg.batch_size = 64
cfg.gpus = '0'
cfg.frequent = 100
cfg.kv_store = 'device'
cfg.memonger = False
cfg.retrain = False
cfg.model_load_epoch = 0
cfg.num_epoch = 100
cfg.model_path = "./model/"
cfg.out_path = "./outputs/"

# network
cfg.network = edict()
cfg.network.ngf = 64
cfg.network.ndf = 64
cfg.network.dropout = False
cfg.network.n_blocks = 9



# Train
cfg.train = edict()
cfg.train.lr = 0.0002
cfg.train.beta1 = 0.5
cfg.train.wd = 0.0


# dataset
cfg.dataset = edict()
cfg.dataset.path = './'
cfg.dataset.h = 256
cfg.dataset.w = 256
cfg.dataset.c = 3
cfg.dataset.dw = 1
cfg.dataset.dh = 1
cfg.dataset.num_pics = 1000
cfg.dataset.num_test = 100



def read_cfg(cfg_file):
    with open(cfg_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in cfg[k]:
                            cfg[k][vk] = vv
                        else:
                            raise ValueError("key {} not exist in config.py".format(vk))
                else:
                    cfg[k] = v
            else:
                raise ValueError("key {} exist in config.py".format(k))