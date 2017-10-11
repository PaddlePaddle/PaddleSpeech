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
from multiprocessing import Process, Manager
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
    :param order: Reserve the order of samples from the given reader.
    :type order: bool
    :return: The wrappered reader
    :rtype: callable
    """
    end_flag = XmapEndSignal()

    # define a worker to read samples from reader to in_queue
    def read_worker(reader, in_queue):
        for sample in reader():
            in_queue.put(sample)
        in_queue.put(end_flag)

    # define a worker to read samples from reader to in_queue with order flag
    def order_read_worker(reader, in_queue):
        for order_id, sample in enumerate(reader()):
            in_queue.put((order_id, sample))
        in_queue.put(end_flag)

    # define a worker to handle samples from in_queue by mapper and put results
    # to out_queue
    def handle_worker(in_queue, out_queue, mapper):
        sample = in_queue.get()
        while not isinstance(sample, XmapEndSignal):
            out_queue.put(mapper(sample))
            sample = in_queue.get()
        in_queue.put(end_flag)
        out_queue.put(end_flag)

    # define a worker to handle samples from in_queue by mapper and put results
    # to out_queue with order
    def order_handle_worker(in_queue, out_queue, mapper, out_order):
        ins = in_queue.get()
        while not isinstance(ins, XmapEndSignal):
            order_id, sample = ins
            result = mapper(sample)
            while order_id != out_order[0]:
                time.sleep(0.001)
            out_queue.put(result)
            out_order[0] += 1
            ins = in_queue.get()
        in_queue.put(end_flag)
        out_queue.put(end_flag)

    # define a thread worker to flush samples from Manager.Queue to Queue
    # for acceleration
    def flush_worker(in_queue, out_queue):
        finish = 0
        while finish < process_num:
            sample = in_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
            else:
                out_queue.put(sample)
        out_queue.put(end_flag)

    def xreader():
        # prepare shared memory
        manager = Manager()
        in_queue = manager.Queue(buffer_size)
        out_queue = manager.Queue(buffer_size)
        out_order = manager.list([0])

        # start a read worker in a process
        target = order_read_worker if order else read_worker
        p = Process(target=target, args=(reader, in_queue))
        p.daemon = True
        p.start()

        # start handle_workers with multiple processes
        target = order_handle_worker if order else handle_worker
        args = (in_queue, out_queue, mapper, out_order) if order else (
            in_queue, out_queue, mapper)
        workers = [
            Process(target=target, args=args) for _ in xrange(process_num)
        ]
        for w in workers:
            w.daemon = True
            w.start()

        # start a thread to read data from slow Manager.Queue
        flush_queue = Queue(buffer_size)
        t = Thread(target=flush_worker, args=(out_queue, flush_queue))
        t.daemon = True
        t.start()

        # get results
        sample = flush_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = flush_queue.get()

    return xreader
