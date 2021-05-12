# -*- coding: utf-8 -*-

import os
import io
import shutil
import codecs
import zipfile

import requests

ROOT = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':

    DOWNLOAD_URL = 'https://cc-cedict.org/editor/editor_export_cedict.php?c=zip'

    zip_file_path = os.path.join(ROOT, 'cc-cedict.zip')

    with open(zip_file_path, 'wb') as f:
        response = requests.get(DOWNLOAD_URL, stream=True)
        shutil.copyfileobj(response.raw, f)

    with open(zip_file_path, 'rb') as fp:
        z = zipfile.ZipFile(fp)
        z.extractall(ROOT)
