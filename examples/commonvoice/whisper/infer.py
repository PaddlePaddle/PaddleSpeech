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
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import yaml

from paddlespeech.s2t.models.whisper.whisper import (CHUNK_LENGTH, MODEL_DIMENSIONS, N_MELS,
                                                   DecodingOptions, Whisper,
                                                   log_mel_spectrogram, pad_or_trim,
                                                   transcribe)
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import get_files_by_ext, str2bool

logger = Log(__name__).getlog()


def load_audio(file_path, sample_rate=16000):
    """Load audio file and convert to 16kHz mono if needed."""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return audio
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        return None


def get_model_path(size, resource_path):
    """Get the model path based on size."""
    return os.path.join(resource_path, "whisper", f"whisper-{size}.pdparams")


def load_whisper_model(model_size="base", checkpoint_path=None, resource_path=None):
    """Load Whisper model from checkpoint or pretrained weights."""
    model_dims = MODEL_DIMENSIONS[model_size]
    model = Whisper(model_dims)
    
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        state_dict = paddle.load(checkpoint_path)
        model.set_state_dict(state_dict)
    elif resource_path:
        model_path = get_model_path(model_size, resource_path)
        if os.path.exists(model_path):
            logger.info(f"Loading pretrained model from: {model_path}")
            state_dict = paddle.load(model_path)
            model.set_state_dict(state_dict)
        else:
            logger.error(f"Pretrained model not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        logger.error("Either checkpoint_path or resource_path must be provided")
        raise ValueError("Either checkpoint_path or resource_path must be provided")
        
    return model


def transcribe_file(
        model,
        audio_file,
        resource_path,
        language=None,
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        without_timestamps=False,
        fp16=False,
        verbose=True,
):
    """Transcribe a single audio file."""
    # Check if file exists
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return None
    
    # Load and process audio
    logger.info(f"Processing audio file: {audio_file}")
    try:
        mel = log_mel_spectrogram(audio_file, resource_path=resource_path)
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None
    
    # Setup decoding options
    decode_options = DecodingOptions(
        language=language,
        task=task,
        temperature=temperature,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        suppress_tokens=suppress_tokens,
        prompt=initial_prompt,
        without_timestamps=without_timestamps,
        fp16=fp16,
    )
    
    # Run transcription
    logger.info("Running transcription...")
    start_time = time.time()
    
    model.eval()
    with paddle.no_grad():
        result = transcribe(
            model=model,
            mel=mel,
            resource_path=resource_path,
            verbose=verbose,
            condition_on_previous_text=condition_on_previous_text,
            **decode_options.__dict__,
        )
    
    elapsed = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed:.2f} seconds")
    
    return result


def batch_transcribe(
        model,
        audio_dir,
        output_dir,
        resource_path,
        extensions=["wav", "mp3", "flac", "m4a", "ogg"],
        **transcribe_kwargs
):
    """Transcribe all audio files in a directory."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(get_files_by_ext(audio_dir, ext))
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir} with extensions {extensions}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    results = {}
    for audio_file in audio_files:
        base_name = os.path.basename(audio_file)
        file_name = os.path.splitext(base_name)[0]
        
        logger.info(f"Processing file: {base_name}")
        result = transcribe_file(model, audio_file, resource_path, **transcribe_kwargs)
        
        if result:
            # Save transcription
            output_file = output_dir / f"{file_name}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"].strip())
            
            # Save detailed results
            results[base_name] = {
                "text": result["text"].strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", ""),
            }
    
    # Save all results as JSON
    import json
    with open(output_dir / "all_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"All transcriptions saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Whisper ASR Inference")
    parser.add_argument("--audio_file", type=str, help="Path to audio file for transcription")
    parser.add_argument("--audio_dir", type=str, help="Path to directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./transcripts", help="Output directory for transcriptions")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint from fine-tuning")
    parser.add_argument("--resource_path", type=str, default="./resources", 
                        help="Path to resources directory containing original models and assets")
    
    # Model options
    parser.add_argument("--use_original", type=str2bool, default=False, 
                        help="Use original Whisper model instead of fine-tuned")
    parser.add_argument("--model_size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        help="Model size for original Whisper")
    
    # Decoding options
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g., en, fr, auto for detection)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe or translate to English")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--best_of", type=int, default=5, help="Number of candidates when sampling with temp > 0")
    parser.add_argument("--patience", type=float, default=1.0, help="Beam search patience factor")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Exponential length penalty")
    parser.add_argument("--suppress_tokens", type=str, default="-1", help="Comma-separated list of token ids to suppress")
    parser.add_argument("--initial_prompt", type=str, default=None, help="Optional text to provide as prompt")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, 
                        help="Whether to condition on previous text")
    parser.add_argument("--without_timestamps", type=str2bool, default=False, help="Don't include timestamps")
    parser.add_argument("--fp16", type=str2bool, default=False, help="Use half-precision float16")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to display the text being decoded")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.audio_file and not args.audio_dir:
        parser.error("Either --audio_file or --audio_dir must be specified")
    
    if args.use_original:
        # Use original Whisper model
        if not args.resource_path:
            parser.error("--resource_path must be specified when using original model")
        
        model = load_whisper_model(
            model_size=args.model_size,
            resource_path=args.resource_path
        )
    else:
        # Use fine-tuned model
        if not args.checkpoint:
            parser.error("--checkpoint must be specified when not using original model")
        
        # Determine model size from checkpoint directory structure
        model_size = "base"  # Default
        if "tiny" in args.checkpoint:
            model_size = "tiny"
        elif "small" in args.checkpoint:
            model_size = "small"
        elif "medium" in args.checkpoint:
            model_size = "medium"
        elif "large" in args.checkpoint:
            model_size = "large"
        
        model = load_whisper_model(
            model_size=model_size,
            checkpoint_path=args.checkpoint,
            resource_path=args.resource_path
        )
    
    # Prepare transcription keyword arguments
    transcribe_kwargs = {
        "language": args.language,
        "task": args.task,
        "temperature": args.temperature,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "suppress_tokens": args.suppress_tokens,
        "initial_prompt": args.initial_prompt,
        "condition_on_previous_text": args.condition_on_previous_text,
        "without_timestamps": args.without_timestamps,
        "fp16": args.fp16,
        "verbose": args.verbose,
    }
    
    # Run transcription
    if args.audio_file:
        # Single file transcription
        result = transcribe_file(model, args.audio_file, args.resource_path, **transcribe_kwargs)
        if result:
            print("-" * 40)
            print("Transcription:")
            print(result["text"])
            print("-" * 40)
            
            # Save to output directory if specified
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                file_name = os.path.splitext(os.path.basename(args.audio_file))[0]
                with open(output_dir / f"{file_name}.txt", "w", encoding="utf-8") as f:
                    f.write(result["text"].strip())
                
                import json
                with open(output_dir / f"{file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"Results saved to {output_dir}")
    
    elif args.audio_dir:
        # Batch transcription
        batch_transcribe(
            model,
            args.audio_dir,
            args.output_dir,
            args.resource_path,
            **transcribe_kwargs
        )


if __name__ == "__main__":
    main()
