"""
Training script for symbolic ontology reasoning.

Trains a small interpretable transformer on symbolic ontology data
for mechanistic interpretability research.
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch


class TrainingLogger:
    """Logger that writes metrics to JSON file for plotting."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        # Clear existing log
        with open(log_file, 'w') as f:
            pass
    
    def log(self, entry: Dict[str, Any]):
        """Append a log entry."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_eval(self, step: int, epoch: int, train_metrics: Dict, 
                 val_metrics: Optional[Dict] = None, lr: float = 0.0):
        """Log evaluation metrics."""
        entry = {
            'type': 'eval',
            'step': step,
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_metrics.get('loss'),
            'train_token_acc': train_metrics.get('token_acc'),
            'train_seq_acc': train_metrics.get('seq_acc'),
        }
        if val_metrics:
            entry['val_loss'] = val_metrics.get('loss')
            entry['val_token_acc'] = val_metrics.get('token_acc')
            entry['val_seq_acc'] = val_metrics.get('seq_acc')
        
        self.log(entry)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from tokenizer import SymbolicOntologyTokenizer
from dataset import SymbolicOntologyDataset, create_dataloaders, collate_fn
from model import InterpretableTransformer, TransformerConfig, count_parameters


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    train_path: str = "symbolic_inductive_train.jsonl"
    val_path: Optional[str] = None
    
    # Model
    n_layers: int = 4
    n_heads: int = 1
    d_model: int = 64
    d_ff: int = 256
    max_seq_len: int = 544  # max_input_len + max_target_len
    concat_pos_emb: bool = True
    pre_ln: bool = True
    use_ff: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 100
    grad_clip: float = 1.0
    
    # Sequence lengths (increased for deep trees)
    max_input_len: int = 512  # Deep trees can have 400+ tokens
    max_target_len: int = 32
    
    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str
) -> torch.Tensor:
    """Compute cross-entropy loss on target tokens only."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]
    
    # Shift logits and labels for next-token prediction
    # logits: (batch, seq, vocab)
    # We want logits[:-1] to predict labels[1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Cross-entropy with label smoothing of 0 (standard CE)
    # -100 labels are ignored
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    
    return loss


def compute_accuracy(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str
) -> Dict[str, float]:
    """
    Compute accuracy metrics.
    
    Returns:
        token_acc: Per-token accuracy on target tokens
        seq_acc: Proportion of sequences with all target tokens correct
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
    
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Shift for comparison
    shift_preds = predictions[:, :-1]
    shift_labels = labels[:, 1:]
    
    # Mask for target tokens (labels != -100)
    target_mask = (shift_labels != -100)
    
    # Token accuracy
    correct_tokens = ((shift_preds == shift_labels) & target_mask).sum().item()
    total_tokens = target_mask.sum().item()
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Sequence accuracy
    # For each sequence, check if all target tokens are correct
    seq_correct = []
    for i in range(shift_labels.size(0)):
        mask_i = target_mask[i]
        if mask_i.sum() == 0:
            seq_correct.append(True)
        else:
            seq_correct.append(
                (shift_preds[i][mask_i] == shift_labels[i][mask_i]).all().item()
            )
    seq_acc = sum(seq_correct) / len(seq_correct)
    
    return {"token_acc": token_acc, "seq_acc": seq_acc}


def evaluate(
    model: nn.Module,
    dataloader,
    device: str,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    
    total_loss = 0.0
    total_token_acc = 0.0
    total_seq_acc = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(model, batch, device)
            acc = compute_accuracy(model, batch, device)
            
            total_loss += loss.item()
            total_token_acc += acc["token_acc"]
            total_seq_acc += acc["seq_acc"]
            n_batches += 1
            
            if max_batches is not None and n_batches >= max_batches:
                break
    
    model.train()
    
    return {
        "loss": total_loss / n_batches,
        "token_acc": total_token_acc / n_batches,
        "seq_acc": total_seq_acc / n_batches
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    checkpoint_dir: str
):
    """Save a training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": asdict(config),
        "epoch": epoch,
        "step": step,
        "metrics": metrics
    }
    
    path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
    
    # Also save latest
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def train(config: TrainingConfig):
    """Main training loop."""
    print("=" * 60)
    print("Training Interpretable Transformer for Ontology Reasoning")
    print("=" * 60)
    
    # Set seed
    set_seed(config.seed)
    
    # Create tokenizer
    tokenizer = SymbolicOntologyTokenizer(max_seq_len=config.max_seq_len)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model config
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=config.max_seq_len,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        pad_token_id=tokenizer.pad_token_id,
        concat_pos_emb=config.concat_pos_emb,
        pre_ln=config.pre_ln,
        use_ff=config.use_ff
    )
    
    # Create model
    model = InterpretableTransformer(model_config)
    model = model.to(config.device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Device: {config.device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config.train_path,
        config.val_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_input_len=config.max_input_len,
        max_target_len=config.max_target_len,
        training_mode="next_token"
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    print(f"Total training steps: {total_steps}")
    print()
    
    # Initialize logger for plotting
    log_file = os.path.join(config.checkpoint_dir, "training_log.json")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(log_file)
    print(f"Logging metrics to: {log_file}")
    print("Run 'python plot_training.py --log-file {log_file}' to visualize")
    print()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            loss = compute_loss(model, batch, config.device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % config.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
            
            # Evaluation
            if global_step % config.eval_interval == 0:
                train_metrics = evaluate(model, train_loader, config.device, max_batches=20)
                print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
                      f"Token Acc: {train_metrics['token_acc']:.2%} | "
                      f"Seq Acc: {train_metrics['seq_acc']:.2%}")
                
                val_metrics = None
                if val_loader:
                    val_metrics = evaluate(model, val_loader, config.device)
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
                          f"Token Acc: {val_metrics['token_acc']:.2%} | "
                          f"Seq Acc: {val_metrics['seq_acc']:.2%}")
                    
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        save_checkpoint(
                            model, optimizer, scheduler, config,
                            epoch, global_step, val_metrics,
                            os.path.join(config.checkpoint_dir, "best")
                        )
                
                # Log for plotting
                lr = scheduler.get_last_lr()[0]
                logger.log_eval(global_step, epoch, train_metrics, val_metrics, lr)
            
            # Checkpointing
            if global_step % config.save_interval == 0:
                metrics = evaluate(model, train_loader, config.device, max_batches=20)
                save_checkpoint(
                    model, optimizer, scheduler, config,
                    epoch, global_step, metrics,
                    config.checkpoint_dir
                )
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{config.num_epochs} complete | "
              f"Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s\n")
    
    # Final save
    metrics = evaluate(model, train_loader, config.device)
    save_checkpoint(
        model, optimizer, scheduler, config,
        config.num_epochs, global_step, metrics,
        config.checkpoint_dir
    )
    
    print("Training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train interpretable transformer")
    
    # Data arguments
    parser.add_argument("--train-path", type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument("--val-path", type=str, default=None,
                       help="Path to validation JSONL file")
    
    # Model arguments
    parser.add_argument("--n-layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=1,
                       help="Number of attention heads")
    parser.add_argument("--d-model", type=int, default=64,
                       help="Model dimension")
    parser.add_argument("--d-ff", type=int, default=256,
                       help="Feedforward dimension")
    parser.add_argument("--no-ff", action="store_true",
                       help="Disable feedforward layers")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-len", type=int, default=512,
                       help="Maximum input sequence length")
    parser.add_argument("--max-target-len", type=int, default=32,
                       help="Maximum target sequence length")
    
    # Other arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create config
    max_seq_len = args.max_input_len + args.max_target_len
    config = TrainingConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_len=max_seq_len,
        use_ff=not args.no_ff,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
