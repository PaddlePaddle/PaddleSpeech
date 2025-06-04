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

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.optimizer import AdamW
from paddle.optimizer.lr import CosineAnnealingDecay, LinearWarmup

from paddlespeech.s2t.models.whisper.tokenizer import get_tokenizer
from paddlespeech.s2t.models.whisper.whisper import MODEL_DIMENSIONS, Whisper
from paddlespeech.s2t.training.reporter import ObsScope, Reporter
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import get_rank

logger = Log(__name__).getlog()


class WhisperTrainer:
    """Trainer for fine-tuning Whisper models."""
    
    def __init__(
            self,
            config: Dict,
            model: Optional[Whisper] = None,
            optimizer: Optional[paddle.optimizer.Optimizer] = None,
            checkpoint_dir: Optional[Union[str, Path]] = None,
            resource_path: Optional[str] = None,
            log_interval: int = 10,
            save_interval: int = 1,
            rank: int = 0,
            world_size: int = 1,
    ):
        """Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
            model: Whisper model instance, or None to build one from config
            optimizer: Optimizer instance, or None to build one from config
            checkpoint_dir: Directory to save checkpoints
            resource_path: Path to resources directory containing whisper assets
            log_interval: Steps between logging
            save_interval: Epochs between checkpoint saving
            rank: Local rank for distributed training
            world_size: Total number of processes for distributed training
        """
        self.config = config
        self.resource_path = resource_path
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.rank = rank
        self.world_size = world_size
        
        # Initialize model if not provided
        self.model = model if model is not None else self._build_model()
        
        # Initialize tokenizer
        self.tokenizer = self._build_tokenizer()
        
        # Initialize optimizer if not provided
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()
        
        # Initialize checkpoint
        self.checkpoint = self._init_checkpoint(checkpoint_dir)
        
        # Initialize reporter and timer
        self.reporter = Reporter()
        self.timer = Timer()
    
    def _build_model(self) -> Whisper:
        """Build Whisper model from config."""
        model_config = self.config['model']
        model_size = model_config['size']
        
        # Load model dimensions
        model_dims = MODEL_DIMENSIONS[model_size]
        
        # Create the model
        model = Whisper(model_dims)
        
        # Load checkpoint if provided
        checkpoint_path = model_config.get('checkpoint')
        if checkpoint_path:
            logger.info(f"Loading model weights from {checkpoint_path}")
            state_dict = paddle.load(checkpoint_path)
            model.set_state_dict(state_dict)
            
        # Freeze encoder if needed
        if model_config.get('freeze_encoder', False):
            logger.info("Freezing encoder parameters")
            for param in model.encoder.parameters():
                param.stop_gradient = True
                
        # Handle distributed training
        if self.world_size > 1:
            logger.info(f"Initializing distributed model with {self.world_size} processes")
            model = paddle.DataParallel(model)
        
        return model
    
    def _build_tokenizer(self):
        """Build Whisper tokenizer."""
        target_language = self.config['data'].get('target_language', 'en')
        task = self.config['data'].get('task', 'transcribe')
        
        return get_tokenizer(
            multilingual=self.model.is_multilingual,
            resource_path=self.resource_path,
            language=target_language,
            task=task
        )
    
    def _build_optimizer(self) -> paddle.optimizer.Optimizer:
        """Build optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Set learning rate
        learning_rate = training_config['learning_rate']
        
        # Apply weight decay
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        
        # Build scheduler
        scheduler_name = training_config.get('scheduler', 'linear')
        max_steps = training_config.get('max_steps', 100000)
        
        # Calculate warmup steps
        warmup_ratio = training_config.get('warmup_ratio', 0.1)
        warmup_steps = int(max_steps * warmup_ratio)
        
        if scheduler_name == 'cosine':
            lr_scheduler = LinearWarmup(
                CosineAnnealingDecay(learning_rate, max_steps - warmup_steps),
                warmup_steps,
                0.0,
                learning_rate
            )
        else:  # default to linear
            lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                paddle.optimizer.lr.PolynomialDecay(
                    learning_rate=learning_rate,
                    decay_steps=max_steps - warmup_steps,
                    end_lr=0.0,
                    power=1.0),
                warmup_steps,
                0.0,
                learning_rate
            )
            
        # Create optimizer
        weight_decay = training_config.get('weight_decay', 0.01)
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        optimizer = AdamW(
            learning_rate=lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            parameters=self.model.parameters(),
            weight_decay=weight_decay,
            grad_clip=nn.ClipGradByNorm(max_grad_norm),
            apply_decay_param_fun=lambda x: x in decay_params
        )
        
        return optimizer
    
    def _init_checkpoint(self, checkpoint_dir=None) -> Checkpoint:
        """Initialize checkpoint for saving and loading."""
        if checkpoint_dir is None:
            checkpoint_dir = self.config['output'].get('checkpoint_dir', './exp/whisper_fine_tune')
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        return Checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            infos=dict(),
            visualizer=None,
            **{"epoch": 0}
        )
    
    def train(self, train_loader: DataLoader, dev_loader: Optional[DataLoader] = None, num_epochs: int = 10):
        """Run training loop.
        
        Args:
            train_loader: DataLoader for training data
            dev_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
        """
        self.reporter.register(self.model, "model")
        
        # Get training configuration
        training_config = self.config['training']
        max_epoch = num_epochs or training_config.get('max_epoch', 10)
        accum_grad = training_config.get('accum_grad', 1)
        
        # Start training
        logger.info(f"Starting training for {max_epoch} epochs")
        self.timer.start()
        
        # Resume from checkpoint if epoch > 0
        start_epoch = self.checkpoint.infos.get("epoch", 0)
        
        for epoch in range(start_epoch, max_epoch):
            self._train_epoch(train_loader, epoch, accum_grad)
            
            # Validation
            if dev_loader is not None:
                dev_loss = self._eval_epoch(dev_loader, epoch)
                self.checkpoint.infos["dev_loss"] = dev_loss
            
            # Update epoch in checkpoint
            self.checkpoint.infos["epoch"] = epoch + 1
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 and self.rank == 0:
                logger.info(f"Saving checkpoint at epoch {epoch + 1}")
                self.checkpoint.save_parameters(tag=f"epoch_{epoch + 1}")
        
        # Save final model
        if self.rank == 0:
            logger.info("Saving final model")
            self.checkpoint.save_parameters(tag="final")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, accum_grad: int = 1):
        """Train for one epoch."""
        self.model.train()
        
        train_loss = 0.0
        num_batches = 0
        steps_per_epoch = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            # Get batch data
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
            train_loss += loss.item() * (accum_grad if accum_grad > 1 else 1)
            num_batches += 1
            
            # Log training progress
            if batch_idx % self.log_interval == 0 and self.rank == 0:
                elapsed_time = self.timer.elapsed_interval()
                step = epoch * steps_per_epoch + batch_idx
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{steps_per_epoch} | "
                            f"Loss: {loss.item()*(accum_grad if accum_grad > 1 else 1):.4f} | "
                            f"Step {step} | {elapsed_time:.2f}s elapsed")
        
        # Compute average loss
        avg_loss = train_loss / num_batches if num_batches > 0 else float('inf')
        
        # Log epoch stats
        if self.rank == 0:
            logger.info(f"Epoch {epoch+1} | Average Training Loss: {avg_loss:.4f}")
            self.checkpoint.infos["train_loss"] = avg_loss
    
    def _eval_epoch(self, dev_loader: DataLoader, epoch: int):
        """Evaluate for one epoch."""
        self.model.eval()
        
        dev_losses = []
        
        with paddle.no_grad():
            for batch in dev_loader:
                if batch is None:
                    continue
                
                # Get batch data
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
        
        avg_loss = sum(dev_losses) / len(dev_losses) if dev_losses else float('inf')
        
        if self.rank == 0:
            logger.info(f"Epoch {epoch+1} | Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save(self, tag: str = "final"):
        """Save model checkpoint."""
        if self.rank == 0:
            logger.info(f"Saving model checkpoint with tag '{tag}'")
            self.checkpoint.save_parameters(tag=tag)
    
    def load(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint.load_parameters(checkpoint_path)
    
    @classmethod
    def from_pretrained(cls, 
                      config: Dict, 
                      model_size: str = "base",
                      checkpoint_path: Optional[str] = None,
                      resource_path: Optional[str] = None):
        """Create a trainer from pretrained model."""
        # Create model
        model_dims = MODEL_DIMENSIONS[model_size]
        model = Whisper(model_dims)
        
        # Load checkpoint if provided
        if checkpoint_path:
            state_dict = paddle.load(checkpoint_path)
            model.set_state_dict(state_dict)
        
        # Create trainer
        trainer = cls(
            config=config,
            model=model,
            resource_path=resource_path
        )
        
        return trainer
