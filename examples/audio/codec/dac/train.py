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
"""Training script for DAC model.

This script demonstrates how to train the DAC model with distributed training support.
"""

import argparse
import os
from pathlib import Path

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
import yaml

from paddlespeech.audio.codec.dac.model import DACModel
from paddlespeech.audio.codec.dac.trainer import DACTrainer
from paddlespeech.audio.codec.dac.processor import DACProcessor
# TODO: Import dataset classes once implemented


def main(args):
    """Main training function.
    
    Args:
        args: Command line arguments
    """
    # Setup distributed training environment
    if args.ngpus > 1:
        dist.init_parallel_env()
        
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    model = DACModel(**config['model'])
    
    if args.ngpus > 1:
        model = paddle.DataParallel(model)
    
    # Setup optimizer
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), 
        learning_rate=config['training']['lr'],
        weight_decay=config['training']['weight_decay'])
    
    # TODO: Setup dataset and dataloader
    # This is a placeholder for the dataset setup
    # train_dataset = ...
    # valid_dataset = ...
    
    # batch_sampler = BatchSampler(...)
    # if args.ngpus > 1:
    #     batch_sampler = DistributedBatchSampler(...)
    
    # train_dataloader = DataLoader(...)
    # valid_dataloader = DataLoader(...)
    
    # Setup trainer
    trainer = DACTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=None,  # TODO: Replace with actual train_dataloader
        output_dir=output_dir,
        config=config,
        max_epoch=args.max_epoch)
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAC model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Directory to save model checkpoints and logs")
    
    parser.add_argument(
        "--ngpus", 
        type=int, 
        default=1,
        help="Number of GPUs for distributed training")
    
    parser.add_argument(
        "--max-epoch", 
        type=int, 
        default=200,
        help="Maximum number of training epochs")
    
    args = parser.parse_args()
    main(args)
