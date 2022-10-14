import json
import sys
import locale
import codecs
import threading
import triton_python_backend_utils as pb_utils

import math
import time
import numpy as np

import onnxruntime as ort

from paddlespeech.server.utils.audio_process import float2pcm
from paddlespeech.server.utils.util import denorm
from paddlespeech.server.utils.util import get_chunks
from paddlespeech.t2s.frontend.zh_frontend import Frontend


voc_block = 36
voc_pad = 14
am_block = 72
am_pad = 12
voc_upsample = 300

# 模型路径
dir_name = "/models/streaming_tts_serving/1"
phones_dict = dir_name + "fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0/phone_id_map.txt"
am_stat_path = dir_name + "fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0/speech_stats.npy"

onnx_am_encoder = dir_name + "fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0/fastspeech2_csmsc_am_encoder_infer.onnx"
onnx_am_decoder = dir_name + "fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0/fastspeech2_csmsc_am_decoder.onnx"
onnx_am_postnet = dir_name + "fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0/fastspeech2_csmsc_am_postnet.onnx"
onnx_voc_melgan = dir_name + "mb_melgan_csmsc_onnx_0.2.0/mb_melgan_csmsc.onnx"

frontend = Frontend(phone_vocab_path=phones_dict, tone_vocab_path=None)
am_mu, am_std = np.load(am_stat_path)

# 用CPU推理
providers = ['CPUExecutionProvider']

# 配置ort session
sess_options = ort.SessionOptions()

# 创建session
am_encoder_infer_sess = ort.InferenceSession(
    onnx_am_encoder, providers=providers, sess_options=sess_options)
am_decoder_sess = ort.InferenceSession(
    onnx_am_decoder, providers=providers, sess_options=sess_options)
am_postnet_sess = ort.InferenceSession(
    onnx_am_postnet, providers=providers, sess_options=sess_options)
voc_melgan_sess = ort.InferenceSession(
    onnx_voc_melgan, providers=providers, sess_options=sess_options)

def depadding(data, chunk_num, chunk_id, block, pad, upsample):
    """
    Streaming inference removes the result of pad inference
    """
    front_pad = min(chunk_id * block, pad)
    # first chunk
    if chunk_id == 0:
        data = data[:block * upsample]
    # last chunk
    elif chunk_id == chunk_num - 1:
        data = data[front_pad * upsample:]
    # middle chunk
    else:
        data = data[front_pad * upsample:(front_pad + block) * upsample]

    return data

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        print(sys.getdefaultencoding())
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)

        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(args['model_name']))

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("input:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("output:", self.output_names)

        # To keep track of response threads so that we can delay
        # the finalizing the model until all response threads
        # have completed.
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        # This model does not support batching, so 'request_count' should always
        # be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException("unsupported batch size " +
                                                len(requests))

        input_data = []
        for idx in range(len(self.input_names)):
            data = pb_utils.get_input_tensor_by_name(requests[0],
                                                     self.input_names[idx])
            data = data.as_numpy()
            data = data[0].decode('utf-8')
            input_data.append(data)
        text = input_data[0]

        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        thread = threading.Thread(target=self.response_thread,
                                  args=(requests[0].get_response_sender(),text)
                                 )
        thread.daemon = True
        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()
        # Unlike in non-decoupled model transaction policy, execute function
        # here returns no response. A return from this function only notifies
        # Triton that the model instance is ready to receive another request. As
        # we are not waiting for the response thread to complete here, it is
        # possible that at any give time the model may be processing multiple
        # requests. Depending upon the request workload, this may lead to a lot
        # of requests being processed by a single model instance at a time. In
        # real-world models, the developer should be mindful of when to return
        # from execute and be willing to accept next request.
        return None

    def response_thread(self, response_sender, text):
        input_ids = frontend.get_input_ids(text, merge_sentences=False, get_tone_ids=False)
        phone_ids = input_ids["phone_ids"]
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i].numpy()
            voc_chunk_id = 0

            orig_hs = am_encoder_infer_sess.run(None, input_feed={'text': part_phone_ids})
            orig_hs = orig_hs[0]

            # streaming voc chunk info
            mel_len = orig_hs.shape[1]
            voc_chunk_num = math.ceil(mel_len / voc_block)
            start = 0
            end = min(voc_block + voc_pad, mel_len)

            # streaming am
            hss = get_chunks(orig_hs, am_block, am_pad, "am")
            am_chunk_num = len(hss)
            for i, hs in enumerate(hss):
                am_decoder_output = am_decoder_sess.run(None, input_feed={'xs': hs})
                am_postnet_output = am_postnet_sess.run(
                    None,
                    input_feed={
                        'xs': np.transpose(am_decoder_output[0], (0, 2, 1))
                    })
                am_output_data = am_decoder_output + np.transpose(am_postnet_output[0], (0, 2, 1))
                normalized_mel = am_output_data[0][0]

                sub_mel = denorm(normalized_mel, am_mu, am_std)
                sub_mel = depadding(sub_mel, am_chunk_num, i, am_block, am_pad, 1)

                if i == 0:
                    mel_streaming = sub_mel
                else:
                    mel_streaming = np.concatenate((mel_streaming, sub_mel), axis=0)


                # streaming voc
                # 当流式AM推理的mel帧数大于流式voc推理的chunk size，开始进行流式voc 推理
                while (mel_streaming.shape[0] >= end and voc_chunk_id < voc_chunk_num):
                    voc_chunk = mel_streaming[start:end, :]

                    sub_wav = voc_melgan_sess.run(
                        output_names=None, input_feed={'logmel': voc_chunk})
                    sub_wav = depadding(sub_wav[0], voc_chunk_num, voc_chunk_id,
                                        voc_block, voc_pad, voc_upsample)

                    output_np = np.array(sub_wav, dtype=self.output_dtype[0])
                    out_tensor1 = pb_utils.Tensor(self.output_names[0], output_np)

                    status = 0 if voc_chunk_id != (voc_chunk_num-1) else 1
                    output_status = np.array([status], dtype=self.output_dtype[1])
                    out_tensor2 = pb_utils.Tensor(self.output_names[1], output_status)

                    inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor1,out_tensor2])

                    #yield sub_wav
                    response_sender.send(inference_response)

                    voc_chunk_id += 1
                    start = max(0, voc_chunk_id * voc_block - voc_pad)
                    end = min((voc_chunk_id + 1) * voc_block + voc_pad, mel_len)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
            
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        Here we will wait for all response threads to complete sending
        responses.
        """
        print('Finalize invoked')

        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = (logging_time_sec / sleep_time_sec)
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = (self.inflight_thread_count != 0)
                if (cycles % cycle_to_log == 0):
                    print(
                        f"Waiting for {self.inflight_thread_count} response threads to complete..."
                    )
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1

        print('Finalize complete...')
