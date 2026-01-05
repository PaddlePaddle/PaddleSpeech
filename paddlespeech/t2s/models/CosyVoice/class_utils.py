import paddle

# from cosyvoice.cli.model import CosyVoice2Model, CosyVoiceModel
# from cosyvoice.flow.flow import CausalMaskedDiffWithXvec, MaskedDiffWithXvec
# from cosyvoice.hifigan.generator import HiFTGenerator
# from cosyvoice.llm.llm import Qwen2LM, TransformerLM
from paddlespeech.t2s.modules.transformer.activation import Swish
from paddlespeech.t2s.modules.transformer.attention import RelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.embedding import EspnetRelPositionalEncoding
from paddlespeech.t2s.modules.transformer.subsampling import LinearNoSubsampling
                                               

COSYVOICE_ACTIVATION_CLASSES = {
    "swish": Swish
}
COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
}
COSYVOICE_EMB_CLASSES = {
    "rel_pos_espnet": EspnetRelPositionalEncoding,
}
COSYVOICE_ATTENTION_CLASSES = {
    "rel_selfattn": RelPositionMultiHeadedAttention,
}


# def get_model_type(configs):
#     if (
#         isinstance(configs["llm"], TransformerLM)
#         and isinstance(configs["flow"], MaskedDiffWithXvec)
#         and isinstance(configs["hift"], HiFTGenerator)
#     ):
#         return CosyVoiceModel
#     if (
#         isinstance(configs["llm"], Qwen2LM)
#         and isinstance(configs["flow"], CausalMaskedDiffWithXvec)
#         and isinstance(configs["hift"], HiFTGenerator)
#     ):
#         return CosyVoice2Model
#     raise TypeError("No valid model type found!")
