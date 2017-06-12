from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.0):
    """Load and parse manifest file."""
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
