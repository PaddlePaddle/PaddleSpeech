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
"""Evaluation script for DAC model.

This script evaluates the DAC model using multiple quality metrics.
"""

import argparse
import os
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm
import yaml

import paddle
from paddle.io import DataLoader

from paddlespeech.audio.codec.dac.model import DACModel
from paddlespeech.audio.codec.dac.evaluator import DACEvaluator, DACAudioMetrics
from paddlespeech.audio.codec.dac.inferencer import DACInferencer
# TODO: Import dataset classes once implemented


def main(args):
    """Run DAC model evaluation on test dataset.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = DACModel(**config['model'])
    
    # Load model checkpoint
    state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(state_dict)
    model.eval()
    
    # Setup distributed evaluation if requested
    if args.ngpus > 1:
        paddle.distributed.init_parallel_env()
        model = paddle.DataParallel(model)
    
    # TODO: Setup dataset and dataloader
    # This is a placeholder for the dataset setup
    # test_dataset = ...
    # test_dataloader = DataLoader(...)
    
    # Initialize evaluator
    evaluator = DACEvaluator(
        model=model,
        dataloader=None,  # TODO: Replace with actual test_dataloader
        sample_rate=config['model'].get('sample_rate', 44100))
    
    # Run evaluation
    results = {}
    if args.mode == 'dataset':
        # Evaluate on full dataset
        results = evaluator.evaluate()
    elif args.mode == 'directory':
        # Evaluate on directory of audio files
        results = evaluate_directory(args.input_dir, args.reference_dir, model, config)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))


def evaluate_directory(input_dir, reference_dir, model, config):
    """Evaluate model on directory of audio files.
    
    Args:
        input_dir: Directory containing input audio files
        reference_dir: Directory containing reference audio files
        model: DAC model instance
        config: Configuration dictionary
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    inferencer = DACInferencer(
        checkpoint_path=None,  # We already loaded the model
        model_config=config['model'])
    inferencer.model = model
    
    metrics_calculator = DACAudioMetrics(
        sample_rate=config['model'].get('sample_rate', 44100))
    
    all_metrics = {}
    file_metrics = []
    
    # Get list of files
    files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    # Process each file
    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        reference_path = os.path.join(reference_dir, filename)
        
        if not os.path.exists(reference_path):
            print(f"Warning: Reference file not found: {reference_path}")
            continue
        
        # Load audio files
        input_audio, sr_in = sf.read(input_path)
        reference_audio, sr_ref = sf.read(reference_path)
        
        # Make mono if stereo
        if input_audio.ndim > 1:
            input_audio = input_audio.mean(axis=1)
        if reference_audio.ndim > 1:
            reference_audio = reference_audio.mean(axis=1)
        
        # Ensure same length
        min_len = min(len(input_audio), len(reference_audio))
        input_audio = input_audio[:min_len]
        reference_audio = reference_audio[:min_len]
        
        # Process through model
        reconstructed_audio = inferencer.reconstruct(input_audio)
        
        # Calculate metrics
        metrics = metrics_calculator.compute_metrics(reference_audio, reconstructed_audio)
        
        # Store per-file results
        file_result = {
            'filename': filename,
            'metrics': metrics
        }
        file_metrics.append(file_result)
        
        # Accumulate metrics
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)
    
    # Calculate averages
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    # Prepare results
    results = {
        'average_metrics': avg_metrics,
        'per_file_metrics': file_metrics
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DAC model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "directory"],
        help="Evaluation mode: use test dataset or directory of files")
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=None,
        help="Directory containing input audio files (for directory mode)")
    
    parser.add_argument(
        "--reference-dir", 
        type=str, 
        default=None,
        help="Directory containing reference audio files (for directory mode)")
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save evaluation results (JSON format)")
    
    parser.add_argument(
        "--ngpus", 
        type=int, 
        default=1,
        help="Number of GPUs for distributed evaluation")
    
    args = parser.parse_args()
    main(args)
