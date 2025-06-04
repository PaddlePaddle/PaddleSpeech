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

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import yaml

from paddlespeech.s2t.models.whisper.tokenizer import get_tokenizer
from paddlespeech.s2t.models.whisper.whisper import (
    MODEL_DIMENSIONS, 
    LANGUAGES, 
    Whisper, 
    DecodingOptions, 
    load_model, 
    transcribe
)
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


def load_whisper_model(
        model_size: str = "base",
        checkpoint_path: Optional[str] = None,
        resource_path: Optional[str] = None,
) -> Whisper:
    """Load Whisper model from checkpoint or pretrained weights.
    
    Args:
        model_size: Model size for Whisper
        checkpoint_path: Path to model checkpoint from fine-tuning
        resource_path: Path to resources directory containing original models
        
    Returns:
        Whisper model
    """
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


def detect_language(
        model: Whisper, 
        mel: paddle.Tensor,
        tokenizer=None,
        resource_path: Optional[str] = None
) -> str:
    """Detect language from audio.
    
    Args:
        model: Whisper model
        mel: Mel spectrogram
        tokenizer: Optional tokenizer
        resource_path: Path to resources directory (required if tokenizer not provided)
        
    Returns:
        Detected language code
    """
    if not tokenizer and not resource_path:
        raise ValueError("Either tokenizer or resource_path must be provided")
        
    if not tokenizer:
        tokenizer = get_tokenizer(
            multilingual=model.is_multilingual,
            resource_path=resource_path
        )
    
    # Get audio features from encoder
    audio_features = model.embed_audio(mel)
    
    # Get initial tokens
    initial_tokens = paddle.to_tensor([[tokenizer.sot]], dtype=paddle.int64)
    
    # Extract language token logits
    token_logits = model.logits(initial_tokens, audio_features)
    language_token_logits = token_logits[:, 0, tokenizer.language_token_ranges]
    
    # Get language token with highest probability
    language_token_probs = paddle.softmax(language_token_logits, axis=-1)
    language_token_id = paddle.argmax(language_token_probs, axis=-1)
    detected_language_token_id = language_token_id.item() + tokenizer.language_token_ranges[0]
    
    # Map token to language code
    language_token = tokenizer.all_tokens[detected_language_token_id]
    language_code = language_token.strip("<>")
    
    return language_code


def get_available_languages() -> List[str]:
    """Get list of available languages in Whisper.
    
    Returns:
        List of language codes
    """
    return sorted(LANGUAGES.keys())


def format_timestamp(
        seconds: float, 
        always_include_hours: bool = False, 
        decimal_marker: str = "."
) -> str:
    """Format a timestamp into a string.
    
    Args:
        seconds: Number of seconds
        always_include_hours: Whether to always include hours
        decimal_marker: Decimal marker character
        
    Returns:
        Formatted timestamp string
    """
    seconds = max(0, seconds)
    hours = seconds // 3600
    seconds = seconds - hours * 3600
    minutes = seconds // 60
    seconds = seconds - minutes * 60
    
    hours_marker = f"{int(hours):02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{int(minutes):02d}:{int(seconds):02d}{decimal_marker}{int(seconds * 100 % 100):02d}"


def save_srt(
        transcript: Dict, 
        file_path: str
):
    """Save transcript to SRT file.
    
    Args:
        transcript: Transcript dictionary from Whisper
        file_path: Path to output SRT file
    """
    if not transcript.get("segments"):
        return
    
    with open(file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcript["segments"], start=1):
            start = format_timestamp(segment["start"], always_include_hours=True, decimal_marker=",")
            end = format_timestamp(segment["end"], always_include_hours=True, decimal_marker=",")
            text = segment["text"].strip().replace("-->", "->")
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def batch_transcribe_with_progress(
        model: Whisper,
        dataset,
        dataloader,
        resource_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        temperature: float = 0.0,
        without_timestamps: bool = True,
        verbose: bool = False,
        decoder_options: Optional[dict] = None,
):
    """Transcribe audio files in batches with progress reporting.
    
    Args:
        model: Whisper model
        dataset: WhisperInferenceDataset
        dataloader: DataLoader from dataset
        resource_path: Path to resources directory
        language: Language code or None for auto-detection
        task: Task (transcribe or translate)
        beam_size: Beam size for beam search
        temperature: Temperature for sampling
        without_timestamps: Whether to include timestamps
        verbose: Whether to show verbose output
        decoder_options: Additional decoder options
        
    Returns:
        List of transcription results
    """
    model.eval()
    tokenizer = get_tokenizer(
        multilingual=model.is_multilingual,
        resource_path=resource_path,
        language=language,
        task=task
    )
    
    results = []
    total_batches = len(dataloader)
    
    with paddle.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose:
                logger.info(f"Processing batch {batch_idx+1}/{total_batches}")
            
            mel = batch["mel"]  # [batch, n_mels, n_frames]
            audio_paths = batch["audio_paths"]
            batch_size = mel.shape[0]
            
            # Process each item in batch
            for i in range(batch_size):
                audio_path = audio_paths[i]
                mel_i = mel[i:i+1]  # Keep batch dimension
                
                # Auto-detect language if needed
                current_language = language
                if not current_language:
                    current_language = detect_language(model, mel_i, tokenizer, resource_path)
                    if verbose:
                        logger.info(f"Detected language: {current_language}")
                
                # Set up decoding options
                options = DecodingOptions(
                    task=task,
                    language=current_language,
                    beam_size=beam_size,
                    temperature=temperature,
                    without_timestamps=without_timestamps,
                    **decoder_options if decoder_options else {}
                )
                
                # Transcribe
                result = transcribe(
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_i,
                    resource_path=resource_path,
                    verbose=verbose,
                    **options.__dict__
                )
                
                # Add file path to result
                result["audio_path"] = audio_path
                results.append(result)
                
                if verbose:
                    logger.info(f"Transcribed {os.path.basename(audio_path)}: {result['text'][:80]}...")
    
    return results
