#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2019 ~ 2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Helper functions to help select gpu automatically.
"""

import subprocess

from sidt import _logger as log


def select_gpus(ngpu, memory=2048):
    """Select gpus
    Select gpu according to free memory

    Args:
        ngpu: number of gpu
        memory: required free memory

    Returns:
        selected_ids: list of available gpu ids
    """

    gpu_info = _get_gpu_status()
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda x: x[1], reverse=True)

    selected_ids = []
    for idx in range(ngpu):
        if sorted_gpu_info[idx][1] > memory:
            selected_ids.append(sorted_gpu_info[idx][0])

    if not selected_ids:
        log.warning("No GPU available.")
    else:
        log.info("Select GPU %s" % (",".join([str(id) for id in selected_ids])))

    return selected_ids


def _get_gpu_status():
    """Get gpu status
    Get gpu status

    Returns:
        gpu_info: a dict, key = gpu_id, value = free memory
    """
    output = subprocess.check_output('nvidia-smi -q --display=MEMORY',
                                     shell=True).decode("utf-8").splitlines()
    gpu_info = dict()
    gpu_id = 0
    for idx, line in enumerate(output):
        line = line.strip()
        if line:
            if line.startswith("GPU"):
                items = [item for item in output[idx + 4].strip().split() if item != ""]
                free_memory = int(items[2])
                gpu_info[gpu_id] = free_memory
                gpu_id += 1

    return gpu_info


if __name__ == "__main__":
    gpus = select_gpus(4)
    gpus = select_gpus(1, 100000)
