# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Inference script for DAC model.

This script demonstrates how to use the DAC model for audio encoding/decoding.
"""

import argparse
import os
import numpy as np
import soundfile as sf
import yaml

import paddle
from paddlespeech.audio.codec.dac.inferencer import DACInferencer


def main(args):
    """Run DAC inference on audio file(s).
    
    Args:
        args: Command line arguments
    """
    # Load configuration if specified
    model_config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            model_config = config.get('model', {})
    
    # Initialize inferencer
    inferencer = DACInferencer(
        checkpoint_path=args.checkpoint,
        model_config=model_config,
        device=args.device)
    
    # Process input file(s)
    if os.path.isfile(args.input):
        process_file(inferencer, args.input, args.output, args.mode)
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, filename)
                process_file(inferencer, input_path, output_path, args.mode)


def process_file(inferencer, input_path, output_path, mode):
    """Process a single audio file.
    
    Args:
        inferencer: DAC inferencer instance
        input_path: Path to input audio file
        output_path: Path to output audio file
        mode: Processing mode (encode, decode, or reconstruct)
    """
    print(f"Processing: {input_path}")
    
    # Load audio
    audio, sample_rate = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if stereo
        
    # Resample if needed
    if sample_rate != inferencer.model.sample_rate:
        # TODO: Implement resampling
        pass
    
    # Process based on mode
    if mode == 'encode':
        # Encode to latent representation
        latent = inferencer.encode(audio)
        # Save latent representation
        np.save(output_path.replace('.wav', '.npy'), latent.numpy())
        
    elif mode == 'decode':
        # Load latent representation
        if input_path.endswith('.npy'):
            latent = paddle.to_tensor(np.load(input_path))
            # Decode from latent representation
            audio_out = inferencer.decode(latent)
            # Save audio
            sf.write(output_path, audio_out, inferencer.model.sample_rate)
        else:
            print(f"Decode mode expects .npy file, got {input_path}")
            
    elif mode == 'reconstruct':
        # Reconstruct audio through encoder-decoder
        audio_out = inferencer.reconstruct(audio)
        # Save reconstructed audio
        sf.write(output_path, audio_out, inferencer.model.sample_rate)
        
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with DAC model")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input audio file or directory of audio files")
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output file or directory")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to model configuration file")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="reconstruct",
        choices=["encode", "decode", "reconstruct"],
        help="Mode: encode (audio to latent), decode (latent to audio), reconstruct (audio to audio)")
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=paddle.get_device(),
        help="Device for inference")
    
    args = parser.parse_args()
    main(args)
