import os
import random
import subprocess

import paddle

NOW_FILE_PATH = os.path.dirname(__file__)
MAIN_ROOT = os.path.realpath(os.path.join(NOW_FILE_PATH, "../../../../"))


def get_ngpu():
    if paddle.device.get_device() == "cpu":
        return 0
    else:
        return 1


def randName(n=5):
    return "".join(random.sample('zyxwvutsrqponmlkjihgfedcba', n))


def SuccessRequest(result=None, message="ok"):
    return {"code": 0, "result": result, "message": message}


def ErrorRequest(result=None, message="error"):
    return {"code": -1, "result": result, "message": message}


def run_cmd(cmd, output_name):
    p = subprocess.Popen(cmd, shell=True)
    res = p.wait()
    print(cmd)
    print("运行结果：", res)
    if res == 0:
        # 运行成功
        if os.path.exists(output_name):
            return output_name
        else:
            # 合成的文件不存在
            return None
    else:
        # 运行失败
        return None
