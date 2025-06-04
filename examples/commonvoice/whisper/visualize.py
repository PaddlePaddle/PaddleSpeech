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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.ticker import MaxNLocator

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


def plot_waveform(audio_path, output_dir=None, show=True):
    """Plot waveform of audio file."""
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Take first channel if stereo
            
        duration = len(audio) / sr
        time_axis = np.linspace(0, duration, len(audio))
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio, color='#1f77b4')
        plt.title(f"Waveform: {os.path.basename(audio_path)}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        
        if output_dir:
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_waveform.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Waveform plot saved to {output_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return True
    except Exception as e:
        logger.error(f"Error plotting waveform: {e}")
        return False


def plot_transcription_comparison(reference, hypothesis, audio_path=None, output_dir=None, show=True):
    """Plot comparison between reference and hypothesis transcriptions."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Title with audio file name if provided
        title = "Reference vs. Hypothesis"
        if audio_path:
            title += f": {os.path.basename(audio_path)}"
        plt.title(title)
        
        # Create comparison table
        rows = []
        rows.append(["Reference", reference])
        rows.append(["Hypothesis", hypothesis])
        
        # Calculate word-level differences
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        
        # Generate color-coded text for differences
        ref_colored = []
        hyp_colored = []
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                ref_colored.extend(ref_words[i1:i2])
                hyp_colored.extend(hyp_words[j1:j2])
            elif op == 'replace':
                ref_colored.extend([f"**{w}**" for w in ref_words[i1:i2]])
                hyp_colored.extend([f"**{w}**" for w in hyp_words[j1:j2]])
            elif op == 'delete':
                ref_colored.extend([f"**{w}**" for w in ref_words[i1:i2]])
            elif op == 'insert':
                hyp_colored.extend([f"**{w}**" for w in hyp_words[j1:j2]])
        
        rows.append(["Ref (Diff)", " ".join(ref_colored)])
        rows.append(["Hyp (Diff)", " ".join(hyp_colored)])
        
        # Calculate error metrics
        word_error = sum(1 for i, j in zip(ref_words, hyp_words) if i != j)
        if len(ref_words) > 0:
            word_error_rate = word_error / len(ref_words)
        else:
            word_error_rate = 0
            
        rows.append(["Word Error", f"{word_error}/{len(ref_words)} ({word_error_rate:.2%})"])
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=rows, colWidths=[0.15, 0.85], loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Highlight difference rows with light blue background
        for i in [2, 3]:
            for j in range(2):
                cell = table._cells[(i, j)]
                cell.set_facecolor('#E6F3FF')
        
        if output_dir:
            base_name = os.path.splitext(os.path.basename(audio_path))[0] if audio_path else "comparison"
            output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return True
    except Exception as e:
        logger.error(f"Error plotting transcription comparison: {e}")
        return False


def plot_evaluation_metrics(results_file, output_dir=None, show=True):
    """Plot evaluation metrics from results file."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        metrics = results['metrics']
        details = results['details']
        
        # Extract duration information
        durations = [item['duration'] for item in details if 'duration' in item]
        if not durations:
            durations = [0] * len(details)  # Default if no durations available
        
        # Calculate per-sample WER
        from paddlespeech.metrics.wer import word_errors
        wers = []
        for item in details:
            ref = item['reference']
            hyp = item['hypothesis']
            word_error_count, word_count = word_errors(ref, hyp)
            wer = word_error_count / max(1, word_count)
            wers.append(wer * 100)  # Convert to percentage
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall metrics
        ax = axes[0, 0]
        metric_names = ['WER', 'CER']
        metric_values = [metrics['wer'] * 100, metrics['cer'] * 100]  # Convert to percentage
        ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e'])
        ax.set_title('Overall Error Metrics')
        ax.set_ylabel('Error Rate (%)')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(metric_values):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        
        # WER vs Duration scatter plot
        ax = axes[0, 1]
        ax.scatter(durations, wers, alpha=0.6)
        ax.set_title('WER vs Audio Duration')
        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('WER (%)')
        ax.grid(alpha=0.3)
        
        # WER distribution histogram
        ax = axes[1, 0]
        ax.hist(wers, bins=20, color='#2ca02c', alpha=0.7)
        ax.set_title('WER Distribution')
        ax.set_xlabel('WER (%)')
        ax.set_ylabel('Number of Samples')
        ax.grid(alpha=0.3)
        
        # Sample count by WER range
        ax = axes[1, 1]
        wer_ranges = [0, 5, 10, 20, 30, 50, 100]
        wer_bins = pd.cut(wers, wer_ranges, right=False)
        wer_counts = pd.value_counts(wer_bins).sort_index()
        
        wer_labels = [f"{wer_ranges[i]}-{wer_ranges[i+1]}%" for i in range(len(wer_ranges)-1)]
        ax.bar(wer_labels, wer_counts, color='#d62728')
        ax.set_title('Sample Count by WER Range')
        ax.set_xlabel('WER Range')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, "evaluation_metrics.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Metrics plots saved to {output_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return True
    except Exception as e:
        logger.error(f"Error plotting evaluation metrics: {e}")
        return False


def visualize_results(results_file, output_dir=None, audio_dir=None, num_samples=5, show=True):
    """Visualize evaluation results."""
    try:
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Plot overall metrics
        plot_evaluation_metrics(results_file, output_dir, show)
        
        # Plot individual examples
        if num_samples > 0:
            details = results['details']
            samples = details[:num_samples]
            
            for i, sample in enumerate(samples):
                audio_path = sample['audio']
                reference = sample['reference']
                hypothesis = sample['hypothesis']
                
                # If audio directory is provided, use audio files from there
                if audio_dir:
                    base_name = os.path.basename(audio_path)
                    audio_path = os.path.join(audio_dir, base_name)
                
                # Plot waveform if audio file exists
                if os.path.exists(audio_path):
                    plot_waveform(audio_path, output_dir, show)
                
                # Plot transcription comparison
                plot_transcription_comparison(reference, hypothesis, audio_path, output_dir, show)
        
        return True
    except Exception as e:
        logger.error(f"Error visualizing results: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Visualize Whisper evaluation results")
    parser.add_argument("--results_file", type=str, required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Directory to save visualization outputs")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing audio files (optional)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of individual samples to visualize")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    visualize_results(
        results_file=args.results_file,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        num_samples=args.num_samples,
        show=args.show
    )


if __name__ == "__main__":
    main()
