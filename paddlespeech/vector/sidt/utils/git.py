#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2019    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Get code information from git
"""

import os
import subprocess


def get_branch():
    """
    Get branch name

    Returns:
        string: branch name
    """
    cur_dir = os.getcwd()
    work_dir = os.path.split(os.path.abspath(__file__))[0]
    os.chdir(work_dir)
    out = subprocess.check_output('git branch | grep "^*"', shell=True)
    out = out.decode("utf8")
    branch = out.split()[-1]
    os.chdir(cur_dir)
    return branch


def get_commit():
    """
    Get commit hash id

    Returns:
        string: commit id
    """
    cur_dir = os.getcwd()
    work_dir = os.path.split(os.path.abspath(__file__))[0]
    os.chdir(work_dir)
    out = subprocess.check_output('git log | head -n 1', shell=True)
    out = out.decode("utf8")
    commit = out.split()[-1]
    os.chdir(cur_dir)
    return commit


def get_unstaged_diff():
    """
    Get unstaged code diff

    Returns:
        string: unstaged code diff
    """
    cur_dir = os.getcwd()
    work_dir = os.path.split(os.path.abspath(__file__))[0]
    os.chdir(work_dir)
    out = subprocess.check_output('git diff', shell=True)
    diff = out.decode("utf8")
    os.chdir(cur_dir)
    return diff


def get_untracked_file(exts=[]):
    """
    Get untracked code file

    Args:
        exts: a list of file extension
    Returns:
        string: content of untracked files, key=filename, value=file content
    """
    cur_dir = os.getcwd()
    work_dir = os.path.split(os.path.abspath(__file__))[0]
    os.chdir(work_dir)
    out = subprocess.check_output('git status -s', shell=True)
    out = out.decode("utf8")
    lines = out.split("\n")
    files = [line.split()[-1] for line in lines if line.startswith("??")]

    contents = {}
    for untracked_file in files:
        _, filename = os.path.split(untracked_file)
        ext = os.path.splitext(filename)[-1]
        if exts and ext not in exts:
            continue
        with open(untracked_file, 'r') as in_fh:
            content = in_fh.read()
            contents[untracked_file] = content

    os.chdir(cur_dir)
    return contents


def get_code_info():
    """
    Get code info, include branch/commit id/code diff

    Returns:
        info string
    """
    code_info = "Code information:\n" + \
                "\tBranch: %s\n" % (get_branch()) + \
                "\tCommit ID: %s\n\n" % (get_commit()) + \
                "-" * 15 + " Code Diff " + "-" * 15 + "\n" + \
                "%s" % (get_unstaged_diff()) + \
                "\n" + "-" * 15 + " Diff End " + "-" * 15 + "\n"
    contents = get_untracked_file([".py", ".sh"])
    for untracked_file, content in contents.items():
        code_info += "\n" + "+" * 10 + "untracked file: %s" % (untracked_file) + "+" * 10 + "\n"
        code_info += "\n" + content + "\n"

    return code_info


if __name__ == "__main__":
    print(get_branch())
    print(get_commit())
    print(get_unstaged_diff())
    print(get_untracked_file([".py"]))
