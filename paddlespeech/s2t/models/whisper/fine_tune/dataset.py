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

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

from paddlespeech.s2t.models.whisper.whisper import (N_MELS, N_SAMPLES, log_mel_spectrogram, pad_or_trim)
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class WhisperDataset(Dataset):
    """Dataset for Whisper fine-tuning"""
    
    def __init__(
            self,
            manifest_path: str,
            tokenizer,
            target_language: str = "en",
            task: str = "transcribe",
            max_duration: float = 30.0,
            min_duration: float = 0.5,
            sample_rate: int = 16000,
            resource_path: str = '',
            pad_to_max_length: bool = False,
    ):
        """Initialize the dataset.
        
        Args:
            manifest_path: Path to manifest file with audio paths and transcripts
            tokenizer: Whisper tokenizer
            target_language: Target language code
            task: Task type, either 'transcribe' or 'translate'
            max_duration: Maximum audio duration
            min_duration: Minimum audio duration
            sample_rate: Audio sample rate
            resource_path: Path to resources directory
            pad_to_max_length: Whether to pad all sequences to maximum length in batch
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.target_language = target_language
        self.task = task
        self.sample_rate = sample_rate
        self.resource_path = resource_path
        self.pad_to_max_length = pad_to_max_length
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf8') as f:
            manifest_lines = f.readlines()
        
        self.data = []
        for line in manifest_lines:
            try:
                item = json.loads(line.strip())
                duration = item.get('duration', 0)
                if min_duration <= duration <= max_duration:
                    self.data.append(item)
            except Exception as e:
                logger.warning(f"Error parsing line in manifest: {e}")
        
        logger.info(f"Loaded {len(self.data)} examples from {manifest_path}")
        
        # Generate special tokens and language tokens
        self.special_tokens = {
            "sot": self.tokenizer.sot,
            "eot": self.tokenizer.eot,
            "translate": self.tokenizer.translate if hasattr(self.tokenizer, "translate") else None,
            "transcribe": self.tokenizer.transcribe if hasattr(self.tokenizer, "transcribe") else None,
            "no_timestamps": self.tokenizer.no_timestamps if hasattr(self.tokenizer, "no_timestamps") else None,
        }
        
        # Get language token
        self.language_token = self.tokenizer.language_tokens.get(self.target_language) if hasattr(self.tokenizer, "language_tokens") else None
        if not self.language_token and hasattr(self.tokenizer, "language_token"):
            self.language_token = self.tokenizer.language_token(self.target_language)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        audio_path = item["audio"]
        try:
            # Process audio to mel spectrogram
            mel = log_mel_spectrogram(audio_path, n_mels=N_MELS, resource_path=self.resource_path)
            mel = pad_or_trim(mel, N_SAMPLES)
            
            # Get text
            text = item["text"]
            
            # Create prompt tokens
            prompt_tokens = [self.special_tokens["sot"]]
            
            # Add language token if available
            if self.language_token is not None:
                prompt_tokens.append(self.language_token)
            
            # Add task token if available
            task_token = self.special_tokens.get(self.task)
            if task_token is not None:
                prompt_tokens.append(task_token)
            
            # Add no_timestamps token if available
            if self.special_tokens["no_timestamps"] is not None:
                prompt_tokens.append(self.special_tokens["no_timestamps"])
            
            # Encode the text
            target_tokens = (
                prompt_tokens + 
                self.tokenizer.encode(text) + 
                [self.special_tokens["eot"]]
            )
            
            return {
                "mel": mel,
                "target_tokens": np.array(target_tokens, dtype=np.int64),
                "text": text,
                "audio_path": audio_path
            }
        
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            # Return a dummy sample that will be filtered in collate_fn
            return None
    
    @staticmethod
    def collate_fn(batch, pad_idx=-100):
        """Collate function for DataLoader"""
        # Filter None samples
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return None
        
        # Get maximum sequence length in this batch
        max_target_len = max(len(sample["target_tokens"]) for sample in batch)
        
        # Initialize tensors
        mel_specs = []
        token_ids = []
        labels = []
        
        # Also collect metadata for debugging/logging
        texts = []
        audio_paths = []
        
        for sample in batch:
            target_tokens = sample["target_tokens"]
            target_len = len(target_tokens)
            
            # Prepare inputs and labels for causal LM training
            # Input tokens are shifted right
            input_tokens = np.zeros(max_target_len, dtype=np.int64)
            input_tokens[:target_len-1] = target_tokens[:target_len-1]  # Exclude EOS
            
            # Labels are shifted left and padded with pad_idx (ignore index)
            label = np.full(max_target_len, pad_idx, dtype=np.int64)
            label[:target_len-1] = target_tokens[1:target_len]  # Start from first token after SOT

            # Add to lists
            mel_specs.append(sample["mel"])
            token_ids.append(input_tokens)
            labels.append(label)
            
            # Collect metadata
            texts.append(sample["text"])
            audio_paths.append(sample["audio_path"])
        
        # Convert to tensors
        mel_specs = paddle.to_tensor(np.array(mel_specs), dtype=paddle.float32)
        token_ids = paddle.to_tensor(np.array(token_ids), dtype=paddle.int64)
        labels = paddle.to_tensor(np.array(labels), dtype=paddle.int64)
        
        return {
            "mel": mel_specs,
            "tokens": token_ids,
            "labels": labels,
            "texts": texts,
            "audio_paths": audio_paths,
        }
    
    def create_dataloader(self, 
                         batch_size=16, 
                         num_workers=4, 
                         shuffle=True, 
                         drop_last=False):
        """Create a dataloader from this dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            drop_last=drop_last
        )


class WhisperInferenceDataset(Dataset):
    """Dataset for Whisper inference with batching support"""
    
    def __init__(
            self,
            audio_paths: Union[str, List[str]],
            tokenizer=None,
            language: Optional[str] = None,
            task: str = "transcribe",
            sample_rate: int = 16000,
            resource_path: str = '',
    ):
        """Initialize the inference dataset.
        
        Args:
            audio_paths: Path to audio file or directory, or list of audio file paths
            tokenizer: Whisper tokenizer (optional, only needed for preparing decoder input)
            language: Language code (e.g., 'en', 'fr')
            task: Task type, either 'transcribe' or 'translate'
            sample_rate: Audio sample rate
            resource_path: Path to resources directory
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.language = language
        self.task = task
        self.sample_rate = sample_rate
        self.resource_path = resource_path
        
        # Process audio paths
        if isinstance(audio_paths, str):
            # Single file
            if os.path.isfile(audio_paths):
                self.audio_paths = [audio_paths]
            # Directory
            elif os.path.isdir(audio_paths):
                self.audio_paths = [
                    os.path.join(audio_paths, f) for f in os.listdir(audio_paths)
                    if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))
                ]
            else:
                raise ValueError(f"Path not found: {audio_paths}")
        else:
            # List of files
            self.audio_paths = audio_paths
            
        logger.info(f"Loaded {len(self.audio_paths)} audio files for inference")
        
        # Generate special tokens and language tokens if tokenizer provided
        self.special_tokens = None
        self.language_token = None
        
        if self.tokenizer:
            self.special_tokens = {
                "sot": self.tokenizer.sot,
                "eot": self.tokenizer.eot,
                "translate": self.tokenizer.translate if hasattr(self.tokenizer, "translate") else None,
                "transcribe": self.tokenizer.transcribe if hasattr(self.tokenizer, "transcribe") else None,
                "no_timestamps": self.tokenizer.no_timestamps if hasattr(self.tokenizer, "no_timestamps") else None,
            }
            
            # Get language token
            if self.language and hasattr(self.tokenizer, "language_tokens"):
                self.language_token = self.tokenizer.language_tokens.get(self.language)
            elif self.language and hasattr(self.tokenizer, "language_token"):
                self.language_token = self.tokenizer.language_token(self.language)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Compute mel spectrogram
        try:
            mel = log_mel_spectrogram(audio_path, self.resource_path, self.sample_rate)
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            # Return zero tensor with correct shape in case of error
            mel = paddle.zeros((N_MELS, N_SAMPLES // 160))
        
        return {
            "mel": mel.numpy(),
            "audio_path": audio_path,
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for inference DataLoader"""
        # Extract mel spectrograms and audio paths
        mel_specs = []
        audio_paths = []
        
        for sample in batch:
            mel_specs.append(sample["mel"])
            audio_paths.append(sample["audio_path"])
        
        # Convert to tensors
        mel_specs = paddle.to_tensor(np.array(mel_specs), dtype=paddle.float32)
        
        return {
            "mel": mel_specs,
            "audio_paths": audio_paths,
        }
    
    def create_dataloader(self, 
                         batch_size=1, 
                         num_workers=4):
        """Create a dataloader from this inference dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            drop_last=False
        )
    
    def prepare_decoder_input(self):
        """Prepare decoder input tokens for initial prompt
        
        Only used when tokenizer is provided.
        """
        if not self.tokenizer:
            logger.warning("Cannot prepare decoder input without tokenizer")
            return None
            
        # Create initial tokens - similar to training but without labels
        tokens = [self.special_tokens["sot"]]
        
        if self.language_token:
            tokens.append(self.language_token)
            
        if self.task == "translate":
            tokens.append(self.special_tokens["translate"])
        elif self.task == "transcribe":
            tokens.append(self.special_tokens["transcribe"])
            
        if self.special_tokens["no_timestamps"]:
            tokens.append(self.special_tokens["no_timestamps"])
            
        return paddle.to_tensor([tokens], dtype=paddle.int64)
