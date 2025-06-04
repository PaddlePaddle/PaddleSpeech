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
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import soundfile
import tqdm
import yaml


def prepare_common_voice(language: str,
                         output_dir: str,
                         cache_dir: Optional[str] = None,
                         val_size: float = 0.03,
                         test_size: float = 0.03,
                         min_duration: float = 0.5,
                         max_duration: float = 30.0,
                         seed: int = 42):
    """
    Prepare Mozilla Common Voice dataset for Whisper fine-tuning.
    
    Args:
        language: Language code (e.g., "en", "fr", "es")
        output_dir: Directory to save preprocessed data
        cache_dir: Cache directory for HuggingFace datasets
        val_size: Validation set size as a fraction of total data
        test_size: Test set size as a fraction of total data
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        seed: Random seed for reproducibility
    """
    print(f"Preparing Common Voice dataset for language: {language}")
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(exist_ok=True)
    
    # Load Common Voice dataset
    print("Loading Common Voice dataset from HuggingFace...")
    try:
        common_voice = datasets.load_dataset(
            "mozilla-foundation/common_voice_11_0", 
            language, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have access to the Common Voice dataset on HuggingFace.")
        return
    
    # Filter and process data
    print("Processing dataset...")
    
    # Function to process and filter examples
    def process_example(example):
        audio = example['audio']
        sample_rate = audio['sampling_rate']
        
        # Calculate duration
        duration = len(audio['array']) / sample_rate
        
        # Filter by duration
        if duration < min_duration or duration > max_duration:
            return None
        
        return {
            "path": None,  # Will be filled later
            "audio": audio,
            "text": example['sentence'],
            "duration": duration,
        }
    
    # Process all splits
    all_data = []
    for split in ['train', 'validation', 'test']:
        if split in common_voice:
            split_data = []
            for example in tqdm.tqdm(common_voice[split], desc=f"Processing {split} set"):
                processed = process_example(example)
                if processed:
                    split_data.append(processed)
            all_data.extend(split_data)
    
    # Shuffle and split data
    random.seed(seed)
    random.shuffle(all_data)
    
    total_size = len(all_data)
    val_count = max(1, int(total_size * val_size))
    test_count = max(1, int(total_size * test_size))
    train_count = total_size - val_count - test_count
    
    train_data = all_data[:train_count]
    val_data = all_data[train_count:train_count + val_count]
    test_data = all_data[train_count + val_count:]
    
    print(f"Dataset split - Train: {len(train_data)}, Dev: {len(val_data)}, Test: {len(test_data)}")
    
    # Save audio files and create manifest files
    def save_manifest(data, name):
        manifest = []
        for i, item in enumerate(tqdm.tqdm(data, desc=f"Saving {name} files")):
            # Generate filename
            filename = f"{name}_{i:08d}.wav"
            filepath = str(output_dir / "wavs" / filename)
            
            # Save audio file
            soundfile.write(
                filepath, 
                item["audio"]["array"], 
                item["audio"]["sampling_rate"]
            )
            
            # Add to manifest
            manifest_item = {
                "utt": f"{name}_{i:08d}",
                "audio": filepath,
                "text": item["text"],
                "duration": item["duration"]
            }
            manifest.append(manifest_item)
        
        # Write manifest file
        with open(output_dir / f"{name}_manifest.json", "w", encoding="utf-8") as f:
            for item in manifest:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    save_manifest(train_data, "train")
    save_manifest(val_data, "dev")
    save_manifest(test_data, "test")
    
    print(f"Dataset preparation complete. Files saved to {output_dir}")
    
    # Save config
    stats = {
        "language": language,
        "total_examples": total_size,
        "train_examples": len(train_data),
        "dev_examples": len(val_data),
        "test_examples": len(test_data),
        "min_duration": min_duration,
        "max_duration": max_duration,
    }
    
    with open(output_dir / "stats.yaml", "w") as f:
        yaml.dump(stats, f)
    
    print("Data preparation complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice dataset for Whisper fine-tuning")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, fr, es)")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save preprocessed data")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace datasets")
    parser.add_argument("--val_size", type=float, default=0.03, help="Validation set size as fraction")
    parser.add_argument("--test_size", type=float, default=0.03, help="Test set size as fraction")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    prepare_common_voice(
        language=args.language,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
