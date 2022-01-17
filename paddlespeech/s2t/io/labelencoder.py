#!/usr/bin/python3
#! coding:utf-8

class LabelEncoder():
    def __init__(self, label_path, label_dataset=[]):
        self.label_path = label_path
        self.update_from_dataset(label_path)

    def encode_sequence(self, sequence, allow_unk=True):
        for label in sequence:
            pass

    def update_from_dataset(self, label_dataset):
        for dataset in label_dataset:
            pass