#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 下午9:49
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : cuda.py
import torch
from fairseq import utils

cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')