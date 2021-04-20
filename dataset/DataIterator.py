#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 下午9:42
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : DataIterator.py
import argparse
from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from utils.seed import seed
from utils.base import get_cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond
        # first call of this method has no effect.
    )
    return batch_iterator

if __name__ == '__main__':
    args = parse_args()
    # todo: 返回是tuble 而不是cfgdic
    hw5_config = get_cfg(args)
    task_cfg = TranslationConfig(
        data=hw5_config.get("data_path"),
        source_lang=hw5_config.get("source_lang"),
        target_lang=hw5_config.get("target_lang"),
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)
    demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
    demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
    sample = next(demo_iter)
    sample