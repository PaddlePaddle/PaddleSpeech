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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
import yaml
from paddle.io import DataLoader, Dataset
from paddle.optimizer import AdamW
from paddle.optimizer.lr import CosineAnnealingDecay, LinearWarmup

from paddlespeech.s2t.models.whisper.tokenizer import get_tokenizer
from paddlespeech.s2t.models.whisper.whisper import (CHUNK_LENGTH, MODEL_DIMENSIONS, N_MELS, N_SAMPLES, Whisper, 
                                                   log_mel_spectrogram, pad_or_trim)
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.training.extensions.evaluator import StandardEvaluator
from paddlespeech.s2t.training.extensions.reporter import ObsScope, Reporter
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import get_rank, str2bool

logger = Log(__name__).getlog()


class WhisperDataset(Dataset):
    """Dataset for Whisper fine-tuning"""
    
    def __init__(
            self,
            manifest_path: str,
            tokenizer,
            target_language: str = "en",
            max_duration: float = 30.0,
            min_duration: float = 0.5,
            sample_rate: int = 16000,
            resource_path: str = '',
    ):
        """Initialize the dataset.
        
        Args:
            manifest_path: Path to manifest file with audio paths and transcripts
            tokenizer: Whisper tokenizer
            target_language: Target language code
            max_duration: Maximum audio duration
            min_duration: Minimum audio duration
            sample_rate: Audio sample rate
            resource_path: Path to resources directory
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.target_language = target_language
        self.sample_rate = sample_rate
        self.resource_path = resource_path
        
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
            "translate": self.tokenizer.translate,
            "transcribe": self.tokenizer.transcribe,
        }
        
        # Get language token
        self.language_token = self.tokenizer.language_tokens[self.target_language]
    
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
            prompt_tokens = [
                self.special_tokens["sot"],
                self.language_token,
                self.special_tokens["transcribe"],
            ]
            
            # Encode the text
            target_tokens = (
                prompt_tokens + 
                self.tokenizer.encode(text) + 
                [self.special_tokens["eot"]]
            )
            
            return {
                "mel": mel,
                "target_tokens": np.array(target_tokens, dtype=np.int64),
                "text": text
            }
        
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            # Return a dummy sample that will be filtered in collate_fn
            return None
    
    @staticmethod
    def collate_fn(batch):
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
        
        for sample in batch:
            target_tokens = sample["target_tokens"]
            target_len = len(target_tokens)
            
            # Prepare inputs and labels for causal LM training
            # Input tokens are shifted right
            input_tokens = np.zeros(max_target_len, dtype=np.int64)
            input_tokens[:target_len-1] = target_tokens[:target_len-1]  # Exclude EOS
            
            # Labels are shifted left and padded with -100 (ignore index)
            label = np.full(max_target_len, -100, dtype=np.int64)
            label[:target_len-1] = target_tokens[1:target_len]  # Start from first token after SOT

            # Add to lists
            mel_specs.append(sample["mel"])
            token_ids.append(input_tokens)
            labels.append(label)
        
        # Convert to tensors
        mel_specs = paddle.to_tensor(np.array(mel_specs), dtype=paddle.float32)
        token_ids = paddle.to_tensor(np.array(token_ids), dtype=paddle.int64)
        labels = paddle.to_tensor(np.array(labels), dtype=paddle.int64)
        
        return {
            "mel": mel_specs,
            "tokens": token_ids,
            "labels": labels,
        }


class WhisperTrainer(Trainer):
    """Trainer for Whisper fine-tuning"""
    
    def __init__(self, config, args):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
            args: Command-line arguments
        """
        super().__init__()
        self.config = config
        self.args = args
        self.resource_path = args.resource_path
        
        # Set random seed
        if args.seed is not None:
            paddle.seed(args.seed)
            np.random.seed(args.seed)
            
        # Initialize distributed training if needed
        if args.distributed:
            dist.init_parallel_env()

        # Build model and optimizer
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        
        # Build tokenizer
        self.tokenizer = get_tokenizer(
            multilingual=self.model.is_multilingual,
            resource_path=self.resource_path,
            language=config['data']['target_language'],
            task="transcribe"
        )
        
        # Initialize checkpoint class
        self._init_checkpoint()
    
    def _build_model(self):
        """Build Whisper model"""
        config = self.config
        model_size = config['model']['size']
        
        # Load model dimensions
        model_dims = MODEL_DIMENSIONS[model_size]
        
        # Create the model
        model = Whisper(model_dims)
        
        # Load checkpoint if provided
        checkpoint_path = config['model']['checkpoint']
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = paddle.load(checkpoint_path)
            model.set_state_dict(state_dict)
            
        # Freeze encoder if needed
        if config['model'].get('freeze_encoder', False):
            logger.info("Freezing encoder parameters")
            for param in model.encoder.parameters():
                param.stop_gradient = True
        
        # Handle distributed training
        if self.args.distributed:
            model = paddle.DataParallel(model)
        
        return model
    
    def _build_optimizer(self):
        """Build optimizer and learning rate scheduler"""
        config = self.config
        model = self.model
        
        # Set learning rate
        learning_rate = config['training']['learning_rate']
        
        # Apply weight decay
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        
        # Build scheduler
        scheduler_name = config['training'].get('scheduler', 'linear')
        num_training_steps = self.config['training'].get('max_steps', 100000)
        
        # Calculate warmup steps
        warmup_ratio = config['training'].get('warmup_ratio', 0.1)
        warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_name == 'cosine':
            lr_scheduler = LinearWarmup(
                CosineAnnealingDecay(learning_rate, num_training_steps - warmup_steps),
                warmup_steps,
                0.0,
                learning_rate
            )
        else:  # default to linear
            lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                paddle.optimizer.lr.PolynomialDecay(
                    learning_rate=learning_rate,
                    decay_steps=num_training_steps - warmup_steps,
                    end_lr=0.0,
                    power=1.0),
                warmup_steps,
                0.0,
                learning_rate
            )
            
        # Create optimizer
        optimizer = AdamW(
            learning_rate=lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            parameters=model.parameters(),
            weight_decay=config['training'].get('weight_decay', 0.01),
            grad_clip=nn.ClipGradByNorm(config['training'].get('max_grad_norm', 1.0)),
            apply_decay_param_fun=lambda x: x in decay_params
        )
        
        return optimizer
    
    def _init_checkpoint(self):
        """Initialize checkpoint for saving and loading"""
        config = self.config
        args = self.args
        
        checkpoint_dir = Path(config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint = Checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            infos=dict(),
            visualizer=None,
            **{"epoch": 0}
        )
        
        # Try to load from checkpoint if provided
        if args.checkpoint_path:
            self.checkpoint.load_parameters(args.checkpoint_path)
    
    def train(self):
        """Run training"""
        config = self.config
        args = self.args
        
        # Create dataset
        train_dataset = WhisperDataset(
            manifest_path=config['data']['train_manifest'],
            tokenizer=self.tokenizer,
            target_language=config['data']['target_language'],
            max_duration=config['data']['max_duration'],
            min_duration=config['data']['min_duration'],
            resource_path=self.resource_path,
        )
        
        dev_dataset = WhisperDataset(
            manifest_path=config['data']['dev_manifest'],
            tokenizer=self.tokenizer,
            target_language=config['data']['target_language'],
            max_duration=config['data']['max_duration'],
            min_duration=config['data']['min_duration'],
            resource_path=self.resource_path,
        )
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4),
            collate_fn=WhisperDataset.collate_fn,
            drop_last=True,
        )
        
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=config['eval']['eval_batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4),
            collate_fn=WhisperDataset.collate_fn,
        )
        
        # Setup training
        max_epoch = config['training']['max_epoch']
        accum_grad = config['training'].get('accum_grad', 1)
        log_interval = config['training'].get('log_interval', 100)
        save_interval = config['output'].get('save_interval', 1)
        
        # Initialize reporter for logging
        reporter = Reporter()
        reporter.register(self.model, "model")
        
        # Initialize evaluator
        evaluator = StandardEvaluator(self.model)
        
        # Training loop
        for epoch in range(max_epoch):
            self.model.train()
            train_loss = 0
            num_batches = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(train_dataloader):
                if batch is None:
                    continue
                
                mel = batch["mel"]
                tokens = batch["tokens"]
                labels = batch["labels"]
                
                # Forward pass
                audio_features = self.model.embed_audio(mel)
                logits = self.model.logits(tokens, audio_features)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape([-1, logits.shape[-1]]),
                    labels.reshape([-1]),
                    ignore_index=-100
                )
                
                # Scale loss for gradient accumulation
                if accum_grad > 1:
                    loss = loss / accum_grad
                
                # Backward pass
                loss.backward()
                
                # Update parameters every accum_grad steps
                if (batch_idx + 1) % accum_grad == 0:
                    self.optimizer.step()
                    self.optimizer.clear_grad()
                
                # Logging
                train_loss += loss.item() * accum_grad
                num_batches += 1
                
                if batch_idx % log_interval == 0 and get_rank() == 0:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Epoch {epoch}/{max_epoch} | Batch {batch_idx}/{len(train_dataloader)} | "
                                f"Loss: {loss.item()*accum_grad:.4f} | {elapsed_time:.2f}s elapsed")
            
            # End of epoch
            avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
            logger.info(f"Epoch {epoch}/{max_epoch} | Average Train Loss: {avg_train_loss:.4f}")
            
            # Evaluation
            self.model.eval()
            dev_losses = []
            
            with paddle.no_grad():
                for batch in dev_dataloader:
                    if batch is None:
                        continue
                    
                    mel = batch["mel"]
                    tokens = batch["tokens"]
                    labels = batch["labels"]
                    
                    # Forward pass
                    audio_features = self.model.embed_audio(mel)
                    logits = self.model.logits(tokens, audio_features)
                    
                    # Compute loss
                    loss = F.cross_entropy(
                        logits.reshape([-1, logits.shape[-1]]),
                        labels.reshape([-1]),
                        ignore_index=-100
                    )
                    
                    dev_losses.append(loss.item())
            
            avg_dev_loss = sum(dev_losses) / len(dev_losses) if dev_losses else float('inf')
            logger.info(f"Epoch {epoch}/{max_epoch} | Validation Loss: {avg_dev_loss:.4f}")
            
            # Update checkpoint info
            self.checkpoint.infos["epoch"] = epoch
            self.checkpoint.infos["train_loss"] = avg_train_loss
            self.checkpoint.infos["dev_loss"] = avg_dev_loss
            
            # Save checkpoint
            if epoch % save_interval == 0 and get_rank() == 0:
                self.checkpoint.save_parameters(tag=f"epoch_{epoch}")
        
        # Save final model
        if get_rank() == 0:
            self.checkpoint.save_parameters(tag="final")
            logger.info(f"Training completed. Final model saved at {self.checkpoint.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Whisper model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resource_path", type=str, default="./resources", help="Path to resources directory")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--distributed", type=str2bool, default=False, help="Enable distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set device
    paddle.set_device(args.device)
    
    trainer = WhisperTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    import json
    main()
