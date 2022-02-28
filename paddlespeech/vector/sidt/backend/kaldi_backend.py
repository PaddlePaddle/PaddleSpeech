#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
config callbacks
"""
from __future__ import print_function
import os
import sys
import numbers
import multiprocessing
import shutil

import paddle
from paddle.hapi.callbacks import ProgBarLogger, ModelCheckpoint, VisualDL
from visualdl import LogWriter

from sidt import _logger as log
from sidt import Trainer
from sidt.utils.gpu import select_gpus
from sidt.utils.seed import seed_everything
import sidt.utils.utils as sidt_utils
import sidt.utils.vdlrecords as sidt_visual


def _train_one_iteration(epoch,
                         rank,
                         gpu_id,
                         ret_queue,
                         **kwargs):
    """
    call the callback function to do one iteration training,
    return the model file and visualdl recording file
    """
    # set env to select gpu
    seed = kwargs['seed'] + rank * 100 + epoch
    seed_everything(seed)
    os.environ["FLAGS_selected_gpus"] = "%d" % gpu_id

    kwargs["save_dir"] = os.path.join(kwargs["save_dir"], "%d" % (rank))
    kwargs["bg_epoch"] = epoch
    kwargs["epochs"] = epoch + 1
    kwargs['seed'] = seed

    # train one iteration
    mdl_dir, log_dir = kwargs['train_func'](**kwargs)

    ret_queue.put((rank, mdl_dir, log_dir))


def _combine_mdl_records(epoch,
                         mdl_dirs,
                         record_dirs,
                         combine_strategy,
                         save_dir):
    """
    combine multiple models and visualdl recording files
    """
    record_files = [sidt_utils.get_latest_file(record_dir) for record_dir in record_dirs]
    mdls = [os.path.join(mdl_dir, "final.pdparams") for mdl_dir in mdl_dirs]
    opt_files = [os.path.join(mdl_dir, "final.pdopt") for mdl_dir in mdl_dirs]

    # combine records
    ret_scalar_dict, valid_idxs = sidt_visual.combine_records(record_files, combine_strategy)

    # combine models
    log.info("Do average for models ...")
    combined_mdl_file = os.path.join(save_dir, 'checkpoint/%d.pdparams' % (epoch))
    combined_opt_file = os.path.join(save_dir, 'checkpoint/%d.pdopt' % (epoch))
    valid_mdls = [mdls[mdl_idx] for mdl_idx in valid_idxs]
    final_state_dict = sidt_utils.avg_models(valid_mdls)
    paddle.save(final_state_dict, combined_mdl_file)
    if os.path.exists(opt_files[0]):
        shutil.copyfile(opt_files[0], combined_opt_file)


    # combine visualdl record files
    log.info("Combine the visualdl record files ...")
    dst_vdlrecord_file = os.path.join(save_dir, 'summary/vdlrecords.log')
    sidt_visual.append_record_file(dst_vdlrecord_file, ret_scalar_dict, final_state_dict)

    log.info("Combine the visualdl record files for each subprocess ...")
    for rank, record_file in enumerate(record_files):
        logdir, _ = os.path.split(record_file)
        dst_vdlrecord_file = os.path.join(logdir, 'vdlrecords.log')
        sidt_visual.concat_records([record_file], dst_vdlrecord_file)

    # delete mdls
    log.info("Delete subprocess mdls and record files ...")
    for mdl_dir in mdl_dirs:
        for name in ["final", "%d" % (epoch)]:
            for ext in ["pdparams", "pdopt"]:
                filepath = os.path.join(mdl_dir, "%s.%s" % (name, ext))
                if os.path.exists(filepath):
                    os.remove(filepath)
    for record_file in record_files:
        os.remove(record_file)

    return combined_mdl_file


def train(train_func,
          save_dir,
          n_procs=1,
          combine_strategy="average",
          batch_size=1,
          eval_batch_size=1,
          bg_epoch=0,
          epochs=1,
          eval_freq=1,
          log_freq=10,
          pretrained_model=None,
          save_freq=1,
          verbose=2,
          drop_last=False,
          shuffle=False,
          num_workers=False,
          seed=0,
          **kwargs
       ):
    """
    Do the training as kaldi recipe.

    Args:
        train_func: callback function to do the model training
        n_procs: number of processes to do one epoch training
        combine_strategy: strategy for combine the models
        other args are same as Trainer.fit function

    Example:
        TODO
    """

    input_kwargs = locals()
    inner_kwargs = input_kwargs['kwargs']
    del input_kwargs['kwargs']
    input_kwargs.update(inner_kwargs)

    assert train_func is not None
    assert combine_strategy in ['average', 'best']

    if bg_epoch >= epochs:
        return

    mp = multiprocessing.get_context("spawn")
    ret_queue = mp.SimpleQueue()

    for epoch in range(bg_epoch, epochs):
        seed_everything(seed + epoch)
        processes = []
        gpu_ids = select_gpus(n_procs)
        assert len(gpu_ids) == n_procs

        # run n_procs jobs
        for rank in range(n_procs):
            process = mp.Process(
                target=_train_one_iteration,
                args=(epoch, rank, gpu_ids[rank], ret_queue),
                kwargs=input_kwargs)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                break

        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()

        # assemble the rets
        mdl_dirs = [None] * n_procs
        record_dirs = [None] * n_procs
        while not ret_queue.empty():
            rank, mdl_dir, record_dir = ret_queue.get()
            mdl_dirs[rank] = mdl_dir
            record_dirs[rank] = record_dir
        assert None not in mdl_dirs and None not in record_dirs

        # combine the models and visual records
        pretrained_model = _combine_mdl_records(epoch, mdl_dirs, record_dirs, combine_strategy, save_dir)
        input_kwargs['pretrained_model'] = pretrained_model

