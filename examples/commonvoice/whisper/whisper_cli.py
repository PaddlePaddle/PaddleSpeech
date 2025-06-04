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
from pathlib import Path

import paddle
import yaml

from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import str2bool

logger = Log(__name__).getlog()


def prepare_parser():
    """Prepare argument parser."""
    parser = argparse.ArgumentParser(
        description="Whisper CLI for fine-tuning, inference, and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data preparation command
    data_parser = subparsers.add_parser("prepare", help="Prepare data for fine-tuning")
    data_parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, fr, es)")
    data_parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save preprocessed data")
    data_parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace datasets")
    data_parser.add_argument("--val_size", type=float, default=0.03, help="Validation set size as fraction")
    data_parser.add_argument("--test_size", type=float, default=0.03, help="Test set size as fraction")
    data_parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration in seconds")
    data_parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration in seconds")
    data_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Fine-tuning command
    train_parser = subparsers.add_parser("train", help="Fine-tune Whisper model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    train_parser.add_argument("--resource_path", type=str, default="./resources", help="Path to resources directory")
    train_parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu"], help="Device to use")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    train_parser.add_argument("--distributed", type=str2bool, default=False, help="Enable distributed training")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Inference with Whisper model")
    infer_parser.add_argument("--audio_file", type=str, help="Path to audio file for transcription")
    infer_parser.add_argument("--audio_dir", type=str, help="Path to directory containing audio files")
    infer_parser.add_argument("--output_dir", type=str, default="./transcripts", help="Output directory for transcriptions")
    infer_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint from fine-tuning")
    infer_parser.add_argument("--resource_path", type=str, default="./resources", 
                        help="Path to resources directory containing original models and assets")
    
    # Model options for inference
    infer_parser.add_argument("--use_original", type=str2bool, default=False, 
                        help="Use original Whisper model instead of fine-tuned")
    infer_parser.add_argument("--model_size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        help="Model size for original Whisper")
    
    # Decoding options for inference
    infer_parser.add_argument("--language", type=str, default=None, help="Language code (e.g., en, fr, auto for detection)")
    infer_parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe or translate to English")
    infer_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    infer_parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    infer_parser.add_argument("--without_timestamps", type=str2bool, default=False, help="Don't include timestamps")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate Whisper model")
    eval_parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    eval_parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint from fine-tuning")
    eval_parser.add_argument("--resource_path", type=str, default="./resources", help="Path to resources directory")
    
    # Model options for evaluation
    eval_parser.add_argument("--model_size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        help="Model size for original Whisper")
    
    # Decoding options for evaluation
    eval_parser.add_argument("--language", type=str, default=None, help="Language code (e.g., en, fr)")
    eval_parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe or translate to English")
    eval_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    eval_parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    eval_parser.add_argument("--without_timestamps", type=str2bool, default=True, help="Don't include timestamps")
    eval_parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == "prepare":
        from prepare_data import prepare_common_voice
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
    
    elif args.command == "train":
        import train
        if args.distributed:
            import paddle.distributed as dist
            dist.init_parallel_env()
        
        # Load configuration
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        # Set device
        paddle.set_device(args.device)
        
        trainer = train.WhisperTrainer(config, args)
        trainer.train()
    
    elif args.command == "infer":
        from infer import load_whisper_model, transcribe_file, batch_transcribe
        
        # Validate arguments
        if not args.audio_file and not args.audio_dir:
            logger.error("Either --audio_file or --audio_dir must be specified")
            sys.exit(1)
        
        if args.use_original:
            # Use original Whisper model
            if not args.resource_path:
                logger.error("--resource_path must be specified when using original model")
                sys.exit(1)
            
            model = load_whisper_model(
                model_size=args.model_size,
                resource_path=args.resource_path
            )
        else:
            # Use fine-tuned model
            if not args.checkpoint:
                logger.error("--checkpoint must be specified when not using original model")
                sys.exit(1)
            
            model = load_whisper_model(
                model_size=args.model_size,
                checkpoint_path=args.checkpoint,
                resource_path=args.resource_path
            )
        
        # Prepare transcription keyword arguments
        transcribe_kwargs = {
            "language": args.language,
            "task": args.task,
            "temperature": args.temperature,
            "beam_size": args.beam_size,
            "without_timestamps": args.without_timestamps,
        }
        
        # Run transcription
        if args.audio_file:
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
            batch_transcribe(
                model,
                args.audio_dir,
                args.output_dir,
                args.resource_path,
                **transcribe_kwargs
            )
    
    elif args.command == "evaluate":
        from evaluate import load_model, evaluate_manifest
        
        model = load_model(
            model_size=args.model_size,
            checkpoint_path=args.checkpoint,
            resource_path=args.resource_path
        )
        
        evaluate_manifest(
            model=model,
            manifest_path=args.manifest,
            resource_path=args.resource_path,
            language=args.language,
            task=args.task,
            temperature=args.temperature,
            beam_size=args.beam_size,
            without_timestamps=args.without_timestamps,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
