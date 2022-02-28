#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
visualdl utilities
"""
import os
import sys
import numpy as np

import visualdl
from visualdl import LogReader

from sidt import _logger as log


def concat_scalars(scalar_list):
    """
    Concat scalars datas

    Args:
        scalar_list: a list of scalar data from LogReader

    Returns:
        ret_data: a  numpy array, shape=(total scalar points, 2)
    """
    assert scalar_list
    n_points = [len(data) for data in scalar_list]
    total = sum(n_points)
    ret_data = np.zeros((total, 2))
    idx = 0
    for data in scalar_list:
        for item in data:
            ret_data[idx][0] = item.timestamp
            ret_data[idx][1] = item.value
            idx += 1

    return ret_data


def avg_scalars(scalar_list):
    """
    Combine scalars datas

    Args:
        scalar_list: a list of scalar data from LogReader

    Returns:
        ret_data: a  numpy array, shape=(length of a scalar data, 2)
    """
    assert scalar_list
    ret_data = np.zeros((len(scalar_list[0]) + 1, 2))
    for data in scalar_list:
        for idx, item in enumerate(data):
            ret_data[idx][0] += item.timestamp
            ret_data[idx][1] += item.value
    ret_data /= len(scalar_list)

    return ret_data


def get_best_index(loss_scalar_list):
    """
    get the scalar index which has best loss

    Args:
        loss_scalar_list: a list of loss scalar data from LogReader

    Returns:
        ret_idx: the index
    """
    assert loss_scalar_list
    ret_idx = -1
    min_loss = 10e30
    for data_idx, data in enumerate(loss_scalar_list):
        values = np.zeros(len(data))
        for idx, item in enumerate(data):
            values[idx] = item.value
        if min_loss > np.mean(values):
            ret_idx = data_idx
            min_loss = np.mean(values)

    assert ret_idx != -1
    return ret_idx


def concat_histograms(histogram_list):
    """
    Concat histograms datas

    Args:
        histogram_list: a list of histogram data from LogReader

    Returns:
        ret_data: a list of tuple(timestamp, point value)
        buckets: buckets of histogram
    """
    assert histogram_list
    ret_data = []
    buckets = 0
    for data in histogram_list:
        for item in data:
            timestamp = item.timestamp
            hist = item.histogram.hist
            bins = item.histogram.bin_edges
            buckets = len(hist)
            values = []
            for idx in range(buckets):
                values.extend([bins[idx]] * int(hist[idx]))
            ret_data.append((timestamp, values))

    return ret_data, buckets


def parse_record_files(record_files):
    """Parse visualdl record files

    Args:
        record_files: record files

    Returns:
        ret_scalar_dict: scalar dict
        ret_histogram_dict: histogram dict
    """
    scalar_dict = {}
    histogram_dict = {}

    for record_file in record_files:
        if not os.path.exists(record_file):
            continue
        reader = LogReader(file_name=record_file)
        tags = reader.tags()

        for tag, data_type in tags.items():
            tag = tag.replace(record_file, "").strip('/')
            dst_tag = tag.replace("%", "/")
            if data_type == "histogram":
                data = reader.get_data('histogram', tag)
                histogram_dict.setdefault(dst_tag, [])
                histogram_dict[dst_tag].append(data)
            elif data_type == "scalar":
                data = reader.get_data('scalar', tag)
                scalar_dict.setdefault(dst_tag, [])
                scalar_dict[dst_tag].append(data)

    return scalar_dict, histogram_dict


def combine_records(record_files, combine_strategy):
    """Combine several visualdl record files

    Args:
        record_files: a list of visualdl record files
        combine_strategy: strategy for combining the record files, value={average, best}

    Returns:
        ret_scalar_dict: a dict of combined scalar data
        valid_idxs: a list of valid record file indexes
    """
    scalar_dict, histogram_dict = parse_record_files(record_files)
    valid_idxs = []
    if combine_strategy == "best":
        valid_idxs.append(get_best_index(scalar_dict['train/loss']))
    else:
        valid_idxs = list(range(len(record_files)))

    ret_scalar_dict = {}
    for tag, scalar_list in scalar_dict.items():
        valid_scalar_list = [scalar_list[idx] for idx in valid_idxs]
        ret_data = avg_scalars(valid_scalar_list)
        ret_scalar_dict[tag] = ret_data

    return ret_scalar_dict, valid_idxs


def concat_records(record_files, out_record_file):
    """Concat several visualdl record files

    Args:
        record_files: a list of visualdl record files
        out_record_file: file path of output visualdl record file
    """
    scalar_dict, histogram_dict = parse_record_files(record_files)
    dst_scalar_dict, dst_histogram_dict = parse_record_files(record_files)

    log_dir, file_name = os.path.split(os.path.abspath(out_record_file))
    with visualdl.LogWriter(logdir=log_dir, file_name=file_name) as writer:
        for tag, scalar_list in scalar_dict.items():
            bg_step = 0 if not dst_scalar_dict else len(dst_scalar_dict[tag][0])
            ret_data = concat_scalars(scalar_list)
            for idx in range(ret_data.shape[0]):
                writer.add_scalar(tag=tag, step=bg_step + idx, value=ret_data[idx][1], walltime=int(ret_data[idx][0]))

        bg_step = 0 if not dst_histogram_dict else len(dst_histogram_dict[list(dst_histogram_dict.keys())[0]][0])
        for tag, histogram_list in histogram_dict.items():
            ret_data, buckets = concat_histograms(histogram_list)
            for idx, (timestamp, values) in enumerate(ret_data):
                writer.add_histogram(tag=tag, step=bg_step + idx, values=values, walltime=timestamp, buckets=buckets)


def append_record_file(record_file, scalar_dict, state_dict):
    """
    Append scalar data and histogram to a record file

    Args:
        record_file: target visualdl record file
        scalar_dict: scalar_dict which generated by combine_records
        state_dict: state dict of paddle layer
    """

    cur_scalar_dict, cur_histogram_dict = parse_record_files([record_file])
    log_dir, file_name = os.path.split(os.path.abspath(record_file))
    with visualdl.LogWriter(logdir=log_dir, file_name=file_name) as writer:
        for tag, scalar_list in scalar_dict.items():
            bg_step = 0 if not cur_scalar_dict else len(cur_scalar_dict[tag][0])
            data = scalar_dict[tag]
            for idx in range(data.shape[0]):
                writer.add_scalar(tag=tag, step=bg_step + idx, value=data[idx][1], walltime=int(data[idx][0]))

        bg_step = 0 if not cur_histogram_dict else len(cur_histogram_dict[list(cur_histogram_dict.keys())[0]][0])
        for name, values in state_dict.items():
            writer.add_histogram(tag='parameters/%s' % (name), values=values.flatten(),
                                 step=bg_step, buckets=10)


if __name__ == "__main__":
    record_files = sys.argv[1:]
    ret_scalar_dict, valid_idxs = combine_records(record_files, "average")
    append_record_file("./temp/vdlrecords.log", ret_scalar_dict, {})

