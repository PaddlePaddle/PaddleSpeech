import json
import logging
import os

import paddle
import torchaudio
import paddlespeech

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def read_lists(list_file):
    lists = []
    with open(list_file, "r", encoding="utf8") as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, "r", encoding="utf8") as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate >= min_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt

    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("failed to parse {}".format(onnx_model))
    for i in range(len(trt_kwargs["input_names"])):
        profile.set_shape(
            trt_kwargs["input_names"][i],
            trt_kwargs["min_shape"][i],
            trt_kwargs["opt_shape"][i],
            trt_kwargs["max_shape"][i],
        )
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


def export_cosyvoice2_vllm(model, model_path, device):
    if os.path.exists(model_path):
        return
    dtype = paddle.bfloat16
    use_bias = True if model.llm_decoder.bias is not None else False
    model.llm.model.lm_head = model.llm_decoder
    embed_tokens = model.llm.model.model.embed_tokens
    model.llm.model.set_input_embeddings(model.speech_embedding)
    model.llm.model.to(device)
    model.llm.model.to(dtype)
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = model.speech_embedding.num_embeddings
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = use_bias
    model.llm.model.save_pretrained(model_path)
    if use_bias is True:
        os.system(
            "sed -i s@Qwen2ForCausalLM@CosyVoice2ForCausalLM@g {}/config.json".format(
                os.path.abspath(model_path)
            )
        )
    model.llm.model.config.vocab_size = tmp_vocab_size
    model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
    model.llm.model.set_input_embeddings(embed_tokens)
