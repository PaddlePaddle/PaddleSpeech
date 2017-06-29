"""Contains data helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


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
    for json_line in open(manifest_path):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if (json_data["duration"] <= max_duration and
                json_data["duration"] >= min_duration):
            manifest.append(json_data)
    return manifest
