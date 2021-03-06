#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 下午9:20
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : setting_v1.py

hw5_config = dict(
    data_path='/Myhome/datasets/lhy2021/HW5/DATA/data-bin/ted2020',
    work_dir='setting_v1',
    source_lang='en',
    target_lang='zh',
    # cpu threads when fetching & processing data.
    num_workers=2,
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,

    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,

    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,

    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,

    # beam size for beam search
    beam=5,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2,
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols.
    post_process="sentencepiece",

    # checkpoints
    keep_last_epochs=5,
    resume=None,  # if resume from checkpoint name (under config.savedir)

    # logging
    use_wandb=False,
)
