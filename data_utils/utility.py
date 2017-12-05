"""Contains data helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
import os
import tarfile
import time
from Queue import Queue
from threading import Thread
from multiprocessing import Process, Manager, Value
from paddle.v2.dataset.common import md5file


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.0):
    """Load and parse manifest file.

    Instances with durations outside [min_duration, max_duration] will be
    filtered out.

    :param manifest_path: Manifest file to load and parse.
    :type manifest_path: basestring
    :param max_duration: Maximal duration in seconds for instance filter.
    :type max_duration: float
    :param min_duration: Minimal duration in seconds for instance filter.
    :type min_duration: float
    :return: Manifest parsing results. List of dict.
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in codecs.open(manifest_path, 'r', 'utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if (json_data["duration"] <= max_duration and
                json_data["duration"] >= min_duration):
            manifest.append(json_data)
    return manifest


def getfile_insensitive(path):
    """Get the actual file path when given insensitive filename."""
    directory, filename = os.path.split(path)
    directory, filename = (directory or '.'), filename.lower()
    for f in os.listdir(directory):
        newpath = os.path.join(directory, f)
        if os.path.isfile(newpath) and f.lower() == filename:
            return newpath


def download_multi(url, target_dir, extra_args):
    """Download multiple files from url to target_dir."""
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    print("Downloading %s ..." % url)
    ret_code = os.system("wget -c " + url + ' ' + extra_args + " -P " +
                         target_dir)
    return ret_code


def download(url, md5sum, target_dir):
    """Download file from url to target_dir, and check md5sum."""
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        os.system("wget -c " + url + " -P " + target_dir)
        print("\nMD5 Chesksum %s ..." % filepath)
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar == True:
        os.remove(filepath)


class XmapEndSignal():
    pass


def xmap_readers_mp(mapper, reader, process_num, buffer_size, order=False):
    """A multiprocessing pipeline wrapper for the data reader.

    :param mapper:  Function to map sample.
    :type mapper: callable
    :param reader: Given data reader.
    :type reader: callable
    :param process_num: Number of processes in the pipeline
    :type process_num: int
    :param buffer_size: Maximal buffer size.
    :type buffer_size: int
    :return: The wrappered reader and cleanup callback
    :rtype: tuple
    """
    end_flag = XmapEndSignal()

    read_workers = []
    handle_workers = []
    flush_workers = []

    read_exit_flag = Value('i', 0)
    handle_exit_flag = Value('i', 0)
    flush_exit_flag = Value('i', 0)

    # define a worker to read samples from reader to in_queue with order flag
    def order_read_worker(reader, in_queue):
        for order_id, sample in enumerate(reader()):
            if read_exit_flag.value == 1: break
            in_queue.put((order_id, sample))
        in_queue.put(end_flag)
        # the reading worker should not exit until all handling work exited
        while handle_exit_flag.value == 0 or read_exit_flag.value == 0:
            time.sleep(0.001)

    # define a worker to handle samples from in_queue by mapper and put results
    # to out_queue with order
    def order_handle_worker(in_queue, out_queue, mapper, out_order):
        ins = in_queue.get()
        while not isinstance(ins, XmapEndSignal):
            if handle_exit_flag.value == 1: break
            order_id, sample = ins
            result = mapper(sample)
            while order_id != out_order[0]:
                time.sleep(0.001)
            out_queue.put(result)
            out_order[0] += 1
            ins = in_queue.get()
        in_queue.put(end_flag)
        out_queue.put(end_flag)
        # wait for exit of flushing worker
        while flush_exit_flag.value == 0 or handle_exit_flag.value == 0:
            time.sleep(0.001)
        read_exit_flag.value = 1
        handle_exit_flag.value = 1

    # define a thread worker to flush samples from Manager.Queue to Queue
    # for acceleration
    def flush_worker(in_queue, out_queue):
        finish = 0
        while finish < process_num and flush_exit_flag.value == 0:
            sample = in_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
            else:
                out_queue.put(sample)
        out_queue.put(end_flag)
        handle_exit_flag.value = 1
        flush_exit_flag.value = 1

    def cleanup():
        # first exit flushing workers
        flush_exit_flag.value = 1
        for w in flush_workers:
            w.join()
        # next exit handling workers
        handle_exit_flag.value = 1
        for w in handle_workers:
            w.join()
        # last exit reading workers
        read_exit_flag.value = 1
        for w in read_workers:
            w.join()

    def xreader():
        # prepare shared memory
        manager = Manager()
        in_queue = manager.Queue(buffer_size)
        out_queue = manager.Queue(buffer_size)
        out_order = manager.list([0])

        # start a read worker in a process
        target = order_read_worker
        p = Process(target=target, args=(reader, in_queue))
        p.daemon = True
        p.start()
        read_workers.append(p)

        # start handle_workers with multiple processes
        target = order_handle_worker
        args = (in_queue, out_queue, mapper, out_order)
        workers = [
            Process(target=target, args=args) for _ in xrange(process_num)
        ]
        for w in workers:
            w.daemon = True
            w.start()
            handle_workers.append(w)

        # start a thread to read data from slow Manager.Queue
        flush_queue = Queue(buffer_size)
        t = Thread(target=flush_worker, args=(out_queue, flush_queue))
        t.daemon = True
        t.start()
        flush_workers.append(t)

        # get results
        sample = flush_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = flush_queue.get()

    return xreader, cleanup
