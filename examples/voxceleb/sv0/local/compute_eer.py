#!/usr/bin/python3
#! coding:utf-8

import sys
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

score_file = sys.argv[1]
labels = []
preds = []

with open(score_file, 'r') as f:
    for line in f:
        label, enroll, test, score = line.strip().split()
        labels.append(int(label))
        preds.append(float(score))

labels = np.array(labels)
preds = np.array(preds)

# Snippet from https://yangcha.github.io/EER-ROC/
fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
print("EER: {}".format(eer))