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
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import yaml

from paddlespeech.s2t.models.whisper.whisper import (MODEL_DIMENSIONS, Whisper, 
                                                    DecodingOptions, log_mel_spectrogram,
                                                    transcribe)
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import get_rank, str2bool
from paddlespeech.metrics.wer import word_errors, char_errors

logger = Log(__name__).getlog()


def compute_metrics(references, hypotheses, language="en"):
    """Compute WER and CER metrics."""
    total_words = 0
    total_chars = 0
    total_word_errors = 0
    total_char_errors = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref = ref.strip()
        hyp = hyp.strip()
        
        word_error_count, word_count = word_errors(ref, hyp, language)
        char_error_count, char_count = char_errors(ref, hyp, language)
        
        total_words += word_count
        total_chars += char_count
        total_word_errors += word_error_count
        total_char_errors += char_error_count
    
    wer = float(total_word_errors) / max(1, total_words)
    cer = float(total_char_errors) / max(1, total_chars)
    
    return {
        "wer": wer,
        "cer": cer,
        "word_errors": total_word_errors,
        "word_count": total_words,
        "char_errors": total_char_errors,
        "char_count": total_chars,
    }


def load_model(model_size="base", checkpoint_path=None, resource_path=None):
    """Load Whisper model from checkpoint or pretrained weights."""
    model_dims = MODEL_DIMENSIONS[model_size]
    model = Whisper(model_dims)
    
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        state_dict = paddle.load(checkpoint_path)
        model.set_state_dict(state_dict)
    elif resource_path:
        model_path = os.path.join(resource_path, "whisper", f"whisper-{model_size}.pdparams")
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


def evaluate_manifest(
        model,
        manifest_path,
        resource_path,
        language=None,
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        patience=1.0,
        batch_size=16,
        without_timestamps=True,
        fp16=False,
        verbose=True,
        output_dir=None,
        max_samples=None
):
    """Evaluate Whisper model on a manifest file."""
    # Load manifest
    with open(manifest_path, 'r', encoding='utf8') as f:
        manifest_lines = f.readlines()
    
    # Limit samples if requested
    if max_samples:
        manifest_lines = manifest_lines[:max_samples]
    
    references = []
    hypotheses = []
    audio_paths = []
    durations = []
    
    # Process each item
    start_time = time.time()
    model.eval()
    
    for i, line in enumerate(manifest_lines):
        if i % 10 == 0:
            logger.info(f"Processing item {i+1}/{len(manifest_lines)}")
        
        item = json.loads(line.strip())
        audio_path = item["audio"]
        reference_text = item["text"]
        
        # Get duration if available
        duration = item.get("duration", 0)
        durations.append(duration)
        
        audio_paths.append(audio_path)
        references.append(reference_text)
        
        # Process audio
        try:
            mel = log_mel_spectrogram(audio_path, resource_path=resource_path)
            
            # Setup decoding options
            decode_options = DecodingOptions(
                language=language,
                task=task,
                temperature=temperature,
                beam_size=beam_size,
                patience=patience,
                without_timestamps=without_timestamps,
                fp16=fp16,
            )
            
            # Run transcription
            with paddle.no_grad():
                result = transcribe(
                    model=model,
                    mel=mel,
                    resource_path=resource_path,
                    verbose=False,
                    **decode_options.__dict__,
                )
            
            hypotheses.append(result["text"])
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            hypotheses.append("")
    
    # Compute metrics
    elapsed = time.time() - start_time
    logger.info(f"Processed {len(manifest_lines)} examples in {elapsed:.2f} seconds")
    
    metrics = compute_metrics(references, hypotheses, language=language if language else "en")
    logger.info(f"WER: {metrics['wer']*100:.2f}%, CER: {metrics['cer']*100:.2f}%")
    
    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results = {
            "metrics": metrics,
            "details": [
                {
                    "audio": audio_path,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "duration": duration
                }
                for audio_path, reference, hypothesis, duration in zip(
                    audio_paths, references, hypotheses, durations
                )
            ]
        }
        
        with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed evaluation results saved to {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper ASR Model")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint from fine-tuning")
    parser.add_argument("--resource_path", type=str, default="./resources", 
                        help="Path to resources directory")
    
    # Model options
    parser.add_argument("--model_size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        help="Model size for original Whisper")
    
    # Decoding options
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g., en, fr)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe or translate to English")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--patience", type=float, default=1.0, help="Beam search patience factor")
    parser.add_argument("--without_timestamps", type=str2bool, default=True, help="Don't include timestamps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--fp16", type=str2bool, default=False, help="Use half-precision float16")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to display verbose logs")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(
        model_size=args.model_size,
        checkpoint_path=args.checkpoint,
        resource_path=args.resource_path
    )
    
    # Evaluate
    evaluate_manifest(
        model=model,
        manifest_path=args.manifest,
        resource_path=args.resource_path,
        language=args.language,
        task=args.task,
        temperature=args.temperature,
        beam_size=args.beam_size,
        patience=args.patience,
        batch_size=args.batch_size,
        without_timestamps=args.without_timestamps,
        fp16=args.fp16,
        verbose=args.verbose,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
