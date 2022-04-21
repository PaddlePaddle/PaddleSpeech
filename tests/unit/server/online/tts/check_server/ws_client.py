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
import _thread as thread
import argparse
import base64
import json
import ssl
import time

import websocket

flag = 1
st = 0.0
all_bytes = b''


class WsParam(object):
    # 初始化
    def __init__(self, text, server="127.0.0.1", port=8090):
        self.server = server
        self.port = port
        self.url = "ws://" + self.server + ":" + str(self.port) + "/ws/tts"
        self.text = text

    # 生成url
    def create_url(self):
        return self.url


def on_message(ws, message):
    global flag
    global st
    global all_bytes

    try:
        message = json.loads(message)
        audio = message["audio"]
        audio = base64.b64decode(audio)  # bytes
        status = message["status"]
        all_bytes += audio

        if status == 0:
            print("create successfully.")
        elif status == 1:
            if flag:
                print(f"首包响应：{time.time() - st} s")
                flag = 0
        elif status == 2:
            final_response = time.time() - st
            duration = len(all_bytes) / 2.0 / 24000
            print(f"尾包响应：{final_response} s")
            print(f"音频时长：{duration} s")
            print(f"RTF: {final_response / duration}")
            with open("./out.pcm", "wb") as f:
                f.write(all_bytes)
            print("ws is closed")
            ws.close()
        else:
            print("infer error")

    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        global st
        text_base64 = str(
            base64.b64encode((wsParam.text).encode('utf-8')), "UTF8")
        d = {"text": text_base64}
        d = json.dumps(d)
        print("Start sending text data")
        st = time.time()
        ws.send(d)

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        help="A sentence to be synthesized",
        default="您好，欢迎使用语音合成服务。")
    parser.add_argument(
        "--server", type=str, help="server ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="server port", default=8092)
    args = parser.parse_args()

    print("***************************************")
    print("Server ip: ", args.server)
    print("Server port: ", args.port)
    print("Sentence to be synthesized: ", args.text)
    print("***************************************")

    wsParam = WsParam(text=args.text, server=args.server, port=args.port)

    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(
        wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
