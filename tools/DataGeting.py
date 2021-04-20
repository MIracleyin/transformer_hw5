#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/17 下午5:28
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : DataGeting.py
from os import system
from pathlib import Path

data_dir = '/Myhome/datasets/lhy2021/HW5/DATA/rawdata'
dataset_name = 'ted2020'
urls = (
    #'"https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%214989&authkey=AGgQ-DaR8eFSl1A"',
    #'"https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%214987&authkey=AA4qP_azsicwZZM"',
    # # If the above links die, use the following instead.
        "https://www.csie.ntu.edu.tw/~r09922057/ML2021-hw5/ted2020.tgz",
        "https://www.csie.ntu.edu.tw/~r09922057/ML2021-hw5/test.tgz",
    # # If the above links die, use the following instead.
    #     "https://mega.nz/#!vEcTCISJ!3Rw0eHTZWPpdHBTbQEqBDikDEdFPr7fI8WxaXK9yZ9U",
    #     "https://mega.nz/#!zNcnGIoJ!oPJX9AvVVs11jc0SaK6vxP_lFUNTkEcK2WbxJpvjU5Y",
)
file_names = (
    'ted2020.tgz',  # train & dev
    'test.tgz',  # test
)
prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)
for u, f in zip(urls, file_names):
    path = prefix / f
    if not path.exists():
        if 'mega' in u:
            system(f"megadl {u} --path {path}")
        else:
            system(f"wget {u} -O {path}")
    if path.suffix == ".tgz":
        system(f"tar -xvf {path} -C {prefix}")
    elif path.suffix == ".zip":
        system(f"unzip -o {path} -d {prefix}")
system(f"mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}")
system(f"mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}")
system(f"mv {prefix/'test.en'} {prefix/'test.raw.en'}")
system(f"mv {prefix/'test.zh'} {prefix/'test.raw.zh'}")
