"""Set up paths for DS2"""

import os.path
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

# Add project path to PYTHONPATH
proj_path = os.path.join(this_dir, '..')
add_path(proj_path)
