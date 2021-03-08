# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tarfile
import zipfile
from paddle.dataset.common import md5file


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


def unzip(filepath, target_dir, rm_tar=False):
    """Unzip the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = zipfile.ZipFile(filepath, 'r')
    tar.extractall(target_dir)
    tar.close()
    if rm_tar == True:
        os.remove(filepath)
