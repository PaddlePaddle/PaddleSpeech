# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
AMI corpus contained 100 hours of meeting recording.
This script returns the standard train, dev and eval split for AMI corpus.
For more information on dataset please refer to http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
"""

ALLOWED_OPTIONS = ["scenario_only", "full_corpus", "full_corpus_asr"]


def get_AMI_split(split_option):
    """
    Prepares train, dev, and test sets for given split_option

    Arguments
    ---------
    split_option: str
        The standard split option.
        Allowed options: "scenario_only", "full_corpus", "full_corpus_asr"

    Returns
    -------
        Meeting IDs for train, dev, and test sets for given split_option
    """

    if split_option not in ALLOWED_OPTIONS:
        print(
            f'Invalid split "{split_option}" requested!\nValid split_options are: ',
            ALLOWED_OPTIONS, )
        return

    if split_option == "scenario_only":

        train_set = [
            "ES2002",
            "ES2005",
            "ES2006",
            "ES2007",
            "ES2008",
            "ES2009",
            "ES2010",
            "ES2012",
            "ES2013",
            "ES2015",
            "ES2016",
            "IS1000",
            "IS1001",
            "IS1002",
            "IS1003",
            "IS1004",
            "IS1005",
            "IS1006",
            "IS1007",
            "TS3005",
            "TS3008",
            "TS3009",
            "TS3010",
            "TS3011",
            "TS3012",
        ]

        dev_set = [
            "ES2003",
            "ES2011",
            "IS1008",
            "TS3004",
            "TS3006",
        ]

        test_set = [
            "ES2004",
            "ES2014",
            "IS1009",
            "TS3003",
            "TS3007",
        ]

    if split_option == "full_corpus":
        # List of train: SA (TRAINING PART OF SEEN DATA)
        train_set = [
            "ES2002",
            "ES2005",
            "ES2006",
            "ES2007",
            "ES2008",
            "ES2009",
            "ES2010",
            "ES2012",
            "ES2013",
            "ES2015",
            "ES2016",
            "IS1000",
            "IS1001",
            "IS1002",
            "IS1003",
            "IS1004",
            "IS1005",
            "IS1006",
            "IS1007",
            "TS3005",
            "TS3008",
            "TS3009",
            "TS3010",
            "TS3011",
            "TS3012",
            "EN2001",
            "EN2003",
            "EN2004",
            "EN2005",
            "EN2006",
            "EN2009",
            "IN1001",
            "IN1002",
            "IN1005",
            "IN1007",
            "IN1008",
            "IN1009",
            "IN1012",
            "IN1013",
            "IN1014",
            "IN1016",
        ]

        # List of dev: SB (DEV PART OF SEEN DATA)
        dev_set = [
            "ES2003",
            "ES2011",
            "IS1008",
            "TS3004",
            "TS3006",
            "IB4001",
            "IB4002",
            "IB4003",
            "IB4004",
            "IB4010",
            "IB4011",
        ]

        # List of test: SC (UNSEEN DATA FOR EVALUATION)
        # Note that IB4005 does not appear because it has speakers in common with two sets of data.
        test_set = [
            "ES2004",
            "ES2014",
            "IS1009",
            "TS3003",
            "TS3007",
            "EN2002",
        ]

    if split_option == "full_corpus_asr":
        train_set = [
            "ES2002",
            "ES2003",
            "ES2005",
            "ES2006",
            "ES2007",
            "ES2008",
            "ES2009",
            "ES2010",
            "ES2012",
            "ES2013",
            "ES2014",
            "ES2015",
            "ES2016",
            "IS1000",
            "IS1001",
            "IS1002",
            "IS1003",
            "IS1004",
            "IS1005",
            "IS1006",
            "IS1007",
            "TS3005",
            "TS3006",
            "TS3007",
            "TS3008",
            "TS3009",
            "TS3010",
            "TS3011",
            "TS3012",
            "EN2001",
            "EN2003",
            "EN2004",
            "EN2005",
            "EN2006",
            "EN2009",
            "IN1001",
            "IN1002",
            "IN1005",
            "IN1007",
            "IN1008",
            "IN1009",
            "IN1012",
            "IN1013",
            "IN1014",
            "IN1016",
        ]

        dev_set = [
            "ES2011",
            "IS1008",
            "TS3004",
            "IB4001",
            "IB4002",
            "IB4003",
            "IB4004",
            "IB4010",
            "IB4011",
        ]

        test_set = [
            "ES2004",
            "IS1009",
            "TS3003",
            "EN2002",
        ]

    return train_set, dev_set, test_set
