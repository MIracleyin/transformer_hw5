#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 下午9:48
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : seed.py
import random
import torch
import numpy as np

seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True