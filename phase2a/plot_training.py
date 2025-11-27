#!/usr/bin/env python3
"""
plot_training.py

Visualization utility for training curves, with grokking detection.

Usage:
    python plot_training.py --log-file training_log.json
    python plot_training.py --log-file training_log.json --output plots/
    python plot_training.py --log-file training_log.json --live  # Auto-refresh
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

# Optional numpy for smoothing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_training_log(log_file: str) -> List[Dict]:
    """Load training log from JSON file."""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def extract_metrics(logs: List[Dict]) -> Dict[str, List]:
    """Extract metric time series from logs."""
    metrics = {
        'step': [],
        'epoch': [],
        'train_loss': [],
        'train_token_acc': [],
        'train_seq_acc': [],
        'val_loss': [],
        'val_token_acc': [],
        'val_seq_acc': [],
        'lr': []
    }
    
    for entry in logs:
        if entry.get('type') == 'eval':
            metrics['step'].append(entry.get('step', 0))
            metrics['epoch'].append(entry.get('epoch', 0))
            metrics['train_loss'].append(entry.get('train_loss'))
            metrics['train_token_acc'].append(entry.get('train_token_acc'))
            metrics['train_seq_acc'].append(entry.get('train_seq_acc'))
            metrics['val_loss'].append(entry.get('val_loss'))
            metrics['val_token_acc'].append(entry.get('val_token_acc'))
            metrics['val_seq_acc'].append(entry.get('val_seq_acc'))
            metrics['lr'].append(entry.get('lr'))
    
    return metrics


def detect_grokking(
    train_acc: List[float], 
    val_acc: List[float],
    train_threshold: float = 0.95,
    val_jump_threshold: float = 0.2,
    window: int = 5
) -> Optional[Tuple[int, str]]:
    """
    Detect grokking phase transition.
    
    Grokking signature:
    1. Training accuracy reaches high level (>95%)
    2. Validation accuracy stays low for a while
    3. Validation accuracy suddenly jumps (>20% increase in short window)
    
    Returns:
        (step_index, description) if grokking detected, None otherwise
    """
    if len(train_acc) < window * 2 or len(val_acc) < window * 2:
        return None
    
    # Filter out None values
    valid_indices = [i for i in range(len(train_acc)) 
                     if train_acc[i] is not None and val_acc[i] is not None]
    
    if len(valid_indices) < window * 2:
        return None
    
    # Find where training accuracy first exceeds threshold
    train_high_idx = None
    for i in valid_indices:
        if train_acc[i] >= train_threshold:
            train_high_idx = i
            break
    
    if train_high_idx is None:
        return None
    
    # Look for sudden jump in validation accuracy after training is high
    for i in range(train_high_idx + window, len(valid_indices) - 1):
        idx = valid_indices[i]
        prev_idx = valid_indices[i - window]
        
        val_now = val_acc[idx]
        val_before = val_acc[prev_idx]
        
        if val_now is not None and val_before is not None:
            jump = val_now - val_before
            if jump >= val_jump_threshold:
                return (idx, f"Val acc jumped {jump:.1%} (from {val_before:.1%} to {val_now:.1%})")
    
    # Check if we're in pre-grokking state (high train, low val)
    if len(valid_indices) > 0:
        last_train = train_acc[valid_indices[-1]]
        last_val = val_acc[valid_indices[-1]]
        if last_train is not None and last_val is not None:
            if last_train >= train_threshold and last_val < 0.5:
                return (-1, f"Pre-grokking: Train={last_train:.1%}, Val={last_val:.1%} - keep training!")
    
    return None


def smooth(values: List[float], window: int = 5) -> List[float]:
    """Apply moving average smoothing."""
    if not HAS_NUMPY:
        return values
    
    # Handle None values
    arr = np.array([v if v is not None else np.nan for v in values])
    
    if len(arr) < window:
        return values
    
    # Simple moving average with nan handling
    smoothed = []
    for i in range(len(arr)):
        start = max(0, i - window // 2)
        end = min(len(arr), i + window // 2 + 1)
        window_vals = arr[start:end]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            smoothed.append(float(np.mean(valid)))
        else:
            smoothed.append(None)
    
    return smoothed


def plot_training_curves(
    metrics: Dict[str, List],
    output_dir: Optional[str] = None,
    show: bool = True,
    smooth_window: int = 3
):
    """Create training visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot without matplotlib. Install with: pip install matplotlib")
        return
    
    steps = metrics['step']
    if not steps:
        print("No evaluation data found in log file.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress - Grokking Analysis', fontsize=14, fontweight='bold')
    
    # Color scheme
    train_color = '#2ecc71'  # Green
    val_color = '#e74c3c'    # Red
    
    # 1. Loss curves (top left)
    ax1 = axes[0, 0]
    train_loss = [v for v in metrics['train_loss'] if v is not None]
    val_loss = [v for v in metrics['val_loss'] if v is not None]
    train_steps = [s for s, v in zip(steps, metrics['train_loss']) if v is not None]
    val_steps = [s for s, v in zip(steps, metrics['val_loss']) if v is not None]
    
    if train_loss:
        ax1.plot(train_steps, train_loss, color=train_color, alpha=0.3, linewidth=1)
        ax1.plot(train_steps, smooth(train_loss, smooth_window), color=train_color, 
                linewidth=2, label='Train Loss')
    if val_loss:
        ax1.plot(val_steps, val_loss, color=val_color, alpha=0.3, linewidth=1)
        ax1.plot(val_steps, smooth(val_loss, smooth_window), color=val_color,
                linewidth=2, label='Val Loss')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Sequence Accuracy (top right) - THE KEY PLOT FOR GROKKING
    ax2 = axes[0, 1]
    train_seq = metrics['train_seq_acc']
    val_seq = metrics['val_seq_acc']
    
    train_seq_valid = [(s, v) for s, v in zip(steps, train_seq) if v is not None]
    val_seq_valid = [(s, v) for s, v in zip(steps, val_seq) if v is not None]
    
    if train_seq_valid:
        t_steps, t_vals = zip(*train_seq_valid)
        ax2.plot(t_steps, t_vals, color=train_color, alpha=0.3, linewidth=1)
        ax2.plot(t_steps, smooth(list(t_vals), smooth_window), color=train_color,
                linewidth=2, label='Train Seq Acc')
    if val_seq_valid:
        v_steps, v_vals = zip(*val_seq_valid)
        ax2.plot(v_steps, v_vals, color=val_color, alpha=0.3, linewidth=1)
        ax2.plot(v_steps, smooth(list(v_vals), smooth_window), color=val_color,
                linewidth=2, label='Val Seq Acc')
    
    # Detect and mark grokking
    grokking = detect_grokking(train_seq, val_seq)
    if grokking:
        grok_idx, grok_msg = grokking
        if grok_idx >= 0 and grok_idx < len(steps):
            ax2.axvline(x=steps[grok_idx], color='purple', linestyle='--', 
                       linewidth=2, label=f'Grokking!')
            ax2.annotate(f'Grokking\n{grok_msg}', 
                        xy=(steps[grok_idx], val_seq[grok_idx]),
                        xytext=(steps[grok_idx] + len(steps)*0.1, 0.5),
                        fontsize=9, color='purple',
                        arrowprops=dict(arrowstyle='->', color='purple'))
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Sequence Accuracy')
    ax2.set_title('Sequence Accuracy (Grokking Indicator)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add grokking status annotation
    if grokking:
        status_color = 'green' if grokking[0] >= 0 else 'orange'
        status_text = "✓ GROKKING DETECTED" if grokking[0] >= 0 else "⏳ " + grokking[1]
    else:
        if train_seq_valid and val_seq_valid:
            last_train = train_seq_valid[-1][1]
            last_val = val_seq_valid[-1][1]
            if last_train > 0.9 and last_val > 0.7:
                status_text = "✓ Good convergence (train & val both high)"
                status_color = 'green'
            elif last_train > 0.9:
                status_text = "⏳ Training memorized, waiting for generalization..."
                status_color = 'orange'
            else:
                status_text = "Training in progress..."
                status_color = 'gray'
        else:
            status_text = "Insufficient data"
            status_color = 'gray'
    
    ax2.text(0.02, 0.98, status_text, transform=ax2.transAxes, 
             fontsize=10, fontweight='bold', color=status_color,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Token Accuracy (bottom left)
    ax3 = axes[1, 0]
    train_tok = metrics['train_token_acc']
    val_tok = metrics['val_token_acc']
    
    train_tok_valid = [(s, v) for s, v in zip(steps, train_tok) if v is not None]
    val_tok_valid = [(s, v) for s, v in zip(steps, val_tok) if v is not None]
    
    if train_tok_valid:
        t_steps, t_vals = zip(*train_tok_valid)
        ax3.plot(t_steps, smooth(list(t_vals), smooth_window), color=train_color,
                linewidth=2, label='Train Token Acc')
    if val_tok_valid:
        v_steps, v_vals = zip(*val_tok_valid)
        ax3.plot(v_steps, smooth(list(v_vals), smooth_window), color=val_color,
                linewidth=2, label='Val Token Acc')
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Token Accuracy')
    ax3.set_title('Token Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)
    
    # 4. Generalization Gap (bottom right)
    ax4 = axes[1, 1]
    
    # Compute gap where both are available
    gaps = []
    gap_steps = []
    for i, s in enumerate(steps):
        if train_seq[i] is not None and val_seq[i] is not None:
            gaps.append(train_seq[i] - val_seq[i])
            gap_steps.append(s)
    
    if gaps:
        # Color by gap magnitude
        colors = ['green' if g < 0.2 else 'orange' if g < 0.5 else 'red' for g in gaps]
        ax4.bar(gap_steps, gaps, color=colors, alpha=0.7, width=max(1, steps[-1]/len(steps)*0.8))
        ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Mild gap (0.2)')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Large gap (0.5)')
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Train - Val Accuracy')
    ax4.set_title('Generalization Gap (Train - Val)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Interpretation guide
    guide_text = "Gap > 0.5: Memorizing\nGap 0.2-0.5: Learning\nGap < 0.2: Generalized"
    ax4.text(0.98, 0.98, guide_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        png_path = output_path / 'training_curves.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {png_path}")
        
        # Save PDF for papers
        pdf_path = output_path / 'training_curves.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot to {pdf_path}")
    
    if show:
        plt.show()
    
    plt.close()


def print_summary(metrics: Dict[str, List]):
    """Print a text summary of training progress."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if not metrics['step']:
        print("No evaluation data found.")
        return
    
    # Latest metrics
    last_idx = -1
    print(f"\nLatest (Step {metrics['step'][last_idx]}, Epoch {metrics['epoch'][last_idx]}):")
    print(f"  Train Loss:     {metrics['train_loss'][last_idx]:.4f}" if metrics['train_loss'][last_idx] else "  Train Loss:     N/A")
    print(f"  Train Seq Acc:  {metrics['train_seq_acc'][last_idx]:.1%}" if metrics['train_seq_acc'][last_idx] else "  Train Seq Acc:  N/A")
    print(f"  Val Loss:       {metrics['val_loss'][last_idx]:.4f}" if metrics['val_loss'][last_idx] else "  Val Loss:       N/A")
    print(f"  Val Seq Acc:    {metrics['val_seq_acc'][last_idx]:.1%}" if metrics['val_seq_acc'][last_idx] else "  Val Seq Acc:    N/A")
    
    # Grokking detection
    print("\n" + "-"*40)
    grokking = detect_grokking(metrics['train_seq_acc'], metrics['val_seq_acc'])
    if grokking:
        grok_idx, grok_msg = grokking
        if grok_idx >= 0:
            print(f"🎉 GROKKING DETECTED at step {metrics['step'][grok_idx]}!")
            print(f"   {grok_msg}")
        else:
            print(f"⏳ {grok_msg}")
    else:
        train_seq = [v for v in metrics['train_seq_acc'] if v is not None]
        val_seq = [v for v in metrics['val_seq_acc'] if v is not None]
        if train_seq and val_seq:
            if train_seq[-1] > 0.9 and val_seq[-1] > 0.7:
                print("✓ Good convergence - both train and val accuracy are high")
            elif train_seq[-1] > 0.9:
                print("⏳ Model has memorized training data, waiting for generalization...")
                print("   Keep training - grokking may still occur!")
            else:
                print("📈 Training in progress...")
    
    # Best validation
    val_seq = metrics['val_seq_acc']
    valid_val = [(i, v) for i, v in enumerate(val_seq) if v is not None]
    if valid_val:
        best_idx, best_val = max(valid_val, key=lambda x: x[1])
        print(f"\nBest Val Seq Acc: {best_val:.1%} at step {metrics['step'][best_idx]}")
    
    print("="*60 + "\n")


def live_plot(log_file: str, output_dir: Optional[str] = None, interval: int = 30):
    """Continuously update plot as training progresses."""
    if not HAS_MATPLOTLIB:
        print("Cannot do live plotting without matplotlib")
        return
    
    print(f"Live plotting from {log_file}")
    print(f"Refreshing every {interval} seconds. Press Ctrl+C to stop.")
    
    import time
    
    plt.ion()  # Interactive mode
    
    try:
        while True:
            try:
                logs = load_training_log(log_file)
                metrics = extract_metrics(logs)
                
                plt.clf()
                plot_training_curves(metrics, output_dir=output_dir, show=False)
                plt.pause(0.1)
                
                print_summary(metrics)
                
            except FileNotFoundError:
                print(f"Waiting for {log_file} to be created...")
            except json.JSONDecodeError:
                print("Log file being written, retrying...")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopped live plotting.")
        plt.ioff()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves with grokking detection")
    parser.add_argument("--log-file", "-l", type=str, default="training_log.json",
                       help="Path to training log JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory for saving plots")
    parser.add_argument("--live", action="store_true",
                       help="Enable live updating plot")
    parser.add_argument("--interval", type=int, default=30,
                       help="Refresh interval for live plotting (seconds)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plot (just save)")
    
    args = parser.parse_args()
    
    if args.live:
        live_plot(args.log_file, args.output, args.interval)
    else:
        try:
            logs = load_training_log(args.log_file)
            metrics = extract_metrics(logs)
            print_summary(metrics)
            plot_training_curves(metrics, output_dir=args.output, show=not args.no_show)
        except FileNotFoundError:
            print(f"Error: Log file '{args.log_file}' not found.")
            print("Make sure training has started and is logging to this file.")
            sys.exit(1)


if __name__ == "__main__":
    main()
