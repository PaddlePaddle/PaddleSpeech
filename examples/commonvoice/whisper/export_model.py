# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
from pathlib import Path

import paddle

from paddlespeech.s2t.models.whisper.whisper import MODEL_DIMENSIONS, Whisper
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


def export_encoder(model, save_path, input_shape=(1, 80, 3000)):
    """Export encoder part of Whisper to inference model."""
    model.eval()
    
    # Create save directory if not exists
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define input spec
    mel_spec = paddle.static.InputSpec(shape=input_shape, dtype='float32', name='mel')
    
    # Export encoder model
    encoder_path = f"{save_path}_encoder"
    paddle.jit.save(
        layer=model.encoder,
        path=encoder_path,
        input_spec=[mel_spec]
    )
    logger.info(f"Encoder model exported to {encoder_path}")
    return encoder_path


def export_decoder(model, save_path, input_shape_tokens=(1, 448), input_shape_features=(1, 1500, 768)):
    """Export decoder part of Whisper to inference model."""
    model.eval()
    
    # Create save directory if not exists
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define input spec
    token_spec = paddle.static.InputSpec(shape=input_shape_tokens, dtype='int64', name='tokens')
    audio_features_spec = paddle.static.InputSpec(shape=input_shape_features, dtype='float32', name='audio_features')
    
    # Create a wrapper to match the exact API of the decoder
    class DecoderWrapper(paddle.nn.Layer):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
        
        def forward(self, tokens, audio_features):
            return self.decoder(tokens, audio_features)
    
    wrapper = DecoderWrapper(model.decoder)
    
    # Export decoder model
    decoder_path = f"{save_path}_decoder"
    paddle.jit.save(
        layer=wrapper,
        path=decoder_path,
        input_spec=[token_spec, audio_features_spec]
    )
    logger.info(f"Decoder model exported to {decoder_path}")
    return decoder_path


def export_whisper(model, save_path):
    """Export full Whisper model to static graph models."""
    export_encoder(model, save_path)
    export_decoder(model, save_path)
    
    # Export model info
    dims = model.dims
    model_info = {
        "n_mels": dims.n_mels,
        "n_vocab": dims.n_vocab,
        "n_audio_ctx": dims.n_audio_ctx,
        "n_audio_state": dims.n_audio_state,
        "n_audio_head": dims.n_audio_head,
        "n_audio_layer": dims.n_audio_layer,
        "n_text_ctx": dims.n_text_ctx,
        "n_text_state": dims.n_text_state,
        "n_text_head": dims.n_text_head,
        "n_text_layer": dims.n_text_layer
    }
    
    # Save model info
    import json
    with open(f"{save_path}_info.json", "w") as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model info saved to {save_path}_info.json")


def main():
    parser = argparse.ArgumentParser(description="Export Whisper model to inference format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save exported model")
    parser.add_argument("--model_size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Model size")
    
    args = parser.parse_args()
    
    # Create model
    model_dims = MODEL_DIMENSIONS[args.model_size]
    model = Whisper(model_dims)
    
    # Load checkpoint
    state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(state_dict)
    
    # Export model
    export_whisper(model, args.output_path)
    logger.info(f"Model exported to {args.output_path}")


if __name__ == "__main__":
    main()
