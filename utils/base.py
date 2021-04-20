#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 下午9:44
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : base.py
import csv
import os
from mmcv import Config

def generate_dir(work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

def get_cfg(args):
    cfg = Config.fromfile(args.config)
    hw5_config = cfg.get('hw5_config')

    work_dir = os.path.join('work_dirs', hw5_config.get('work_dir'))
    model_dir = os.path.join(work_dir, 'models')
    return hw5_config, work_dir, model_dir
