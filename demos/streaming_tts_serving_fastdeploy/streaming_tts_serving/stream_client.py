#!/usr/bin/env python
import argparse
import queue
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

FLAGS = None


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def async_stream_send(triton_client, values, request_id, model_name):

    infer_inputs = []
    outputs = []
    for idx, data in enumerate(values):
        data = np.array([data.encode('utf-8')], dtype=np.object_)
        infer_input = grpcclient.InferInput('INPUT_0', [len(data)], "BYTES")
        infer_input.set_data_from_numpy(data)
        infer_inputs.append(infer_input)

        outputs.append(grpcclient.InferRequestedOutput('OUTPUT_0'))
        # Issue the asynchronous sequence inference.
        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=infer_inputs,
            outputs=outputs,
            request_id=request_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action="store_true",
        required=False,
        default=False,
        help='Enable verbose output')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8001',
        help='Inference server URL and it gRPC port. Default is localhost:8001.')

    FLAGS = parser.parse_args()

    # We use custom "sequence" models which take 1 input
    # value. The output is the accumulated value of the inputs. See
    # src/custom/sequence.
    model_name = "streaming_tts_serving"

    values = ["哈哈哈哈"]

    request_id = "0"

    string_result0_list = []

    user_data = UserData()

    # It is advisable to use client object within with..as clause
    # when sending streaming requests. This ensures the client
    # is closed when the block inside with exits.
    with grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
        try:
            # Establish stream
            triton_client.start_stream(callback=partial(callback, user_data))
            # Now send the inference sequences...
            async_stream_send(triton_client, values, request_id, model_name)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

        # Retrieve results...
        recv_count = 0
        result_dict = {}
        status = True
        while True:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                raise data_item
            else:
                this_id = data_item.get_response().id
                if this_id not in result_dict.keys():
                    result_dict[this_id] = []
                result_dict[this_id].append((recv_count, data_item))
                sub_wav = data_item.as_numpy('OUTPUT_0')
                status = data_item.as_numpy('status')
                print('sub_wav = ', sub_wav, "subwav.shape = ", sub_wav.shape)
                print('status = ', status)
                if status[0] == 1:
                    break
            recv_count += 1

    print("PASS: stream_client")
