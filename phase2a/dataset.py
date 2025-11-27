"""
PyTorch Dataset for symbolic ontology reasoning.

Reads JSONL files produced by generate_symbolic_ontology.py and
prepares them for training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path

from tokenizer import SymbolicOntologyTokenizer


class SymbolicOntologyDataset(Dataset):
    """
    Dataset for symbolic ontology reasoning tasks.
    
    Each sample consists of:
    - input: World model + observations + task prompt
    - target: The hypothesis to predict
    - metadata: Additional info for MI analysis
    """
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: SymbolicOntologyTokenizer,
        max_input_len: int = 256,
        max_target_len: int = 32,
        training_mode: str = "next_token",  # "next_token" or "seq2seq"
    ):
        """
        Initialize dataset.
        
        Args:
            jsonl_path: Path to JSONL file
            tokenizer: SymbolicOntologyTokenizer instance
            max_input_len: Maximum input sequence length
            max_target_len: Maximum target sequence length
            training_mode: "next_token" concatenates input+target for autoregressive training
                          "seq2seq" keeps them separate
        """
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.training_mode = training_mode
        
        self.samples = []
        self._load_data(jsonl_path)
        
    def _load_data(self, jsonl_path: str):
        """Load samples from JSONL file."""
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns dict with:
            - input_ids: Token IDs for input (+ target if next_token mode)
            - labels: Target token IDs (shifted for next_token mode)
            - attention_mask: 1s for real tokens, 0s for padding
            - metadata: Original metadata dict (for MI analysis)
        """
        sample = self.samples[idx]
        
        # Encode input
        input_text = sample["input"]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        
        # Encode target
        target_text = sample["target"]
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        # Add EOS to target
        target_ids = target_ids + [self.tokenizer.eos_token_id]
        
        if self.training_mode == "next_token":
            # Concatenate input + [ANSWER] + target for autoregressive training
            answer_token_id = self.tokenizer.token_to_id[self.tokenizer.ANSWER_TOKEN]
            newline_id = self.tokenizer.token_to_id["\n"]
            
            # Remove EOS from input (we'll add it at the end)
            if input_ids[-1] == self.tokenizer.eos_token_id:
                input_ids = input_ids[:-1]
            
            # IMPORTANT: Truncate input and target BEFORE concatenation
            # Reserve space for [ANSWER], \n, and some target tokens
            max_input_tokens = self.max_input_len - 2  # Leave room for [ANSWER] and \n
            if len(input_ids) > max_input_tokens:
                input_ids = input_ids[:max_input_tokens]
            
            if len(target_ids) > self.max_target_len:
                target_ids = target_ids[:self.max_target_len]
            
            # Full sequence: input + [ANSWER] + target + EOS
            full_ids = input_ids + [answer_token_id, newline_id] + target_ids
            
            # Calculate max_len and ensure consistency
            max_len = self.max_input_len + self.max_target_len
            
            # Final truncation (safety)
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]
            
            # Find where the target starts (after [ANSWER]\n) BEFORE padding
            target_start = len(input_ids) + 2  # +2 for [ANSWER] and \n
            
            # Pad to max_len
            attention_mask = [1] * len(full_ids)
            padding_len = max_len - len(full_ids)
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            
            # Labels: same as input but shifted (predict next token)
            # We mask out the input portion so loss is only computed on target
            labels = full_ids.copy()
            # Mask input portion with -100 (ignored by CrossEntropyLoss)
            labels[:target_start] = [-100] * target_start
            
            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "metadata": sample.get("metadata", {}),
                "sample_id": sample.get("id", idx)
            }
            
        else:  # seq2seq mode
            # Pad input
            input_attention_mask = [1] * len(input_ids)
            if len(input_ids) > self.max_input_len:
                input_ids = input_ids[:self.max_input_len]
                input_attention_mask = input_attention_mask[:self.max_input_len]
            else:
                padding_len = self.max_input_len - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
                input_attention_mask = input_attention_mask + [0] * padding_len
            
            # Pad target
            if len(target_ids) > self.max_target_len:
                target_ids = target_ids[:self.max_target_len]
            else:
                padding_len = self.max_target_len - len(target_ids)
                target_ids = target_ids + [self.tokenizer.pad_token_id] * padding_len
            
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(target_ids, dtype=torch.long),
                "attention_mask": torch.tensor(input_attention_mask, dtype=torch.long),
                "metadata": sample.get("metadata", {}),
                "sample_id": sample.get("id", idx)
            }
    
    def get_metadata_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset for analysis."""
        stats = {
            "total_samples": len(self.samples),
            "task_types": {},
            "parsimony_tests": 0,
            "depth_distribution": {},
            "tree_depth_distribution": {},
        }
        
        for sample in self.samples:
            metadata = sample.get("metadata", {})
            
            # Task type distribution
            task_type = sample.get("task_type", "unknown")
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
            
            # Parsimony test count
            if metadata.get("is_parsimony_test", False):
                stats["parsimony_tests"] += 1
            
            # Depth of truth distribution
            depth = metadata.get("depth_of_truth", -1)
            stats["depth_distribution"][depth] = stats["depth_distribution"].get(depth, 0) + 1
            
            # Tree depth distribution
            tree_depth = metadata.get("tree_depth", -1)
            stats["tree_depth_distribution"][tree_depth] = \
                stats["tree_depth_distribution"].get(tree_depth, 0) + 1
        
        stats["parsimony_test_ratio"] = stats["parsimony_tests"] / len(self.samples)
        
        return stats


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function that handles metadata.
    
    Returns a batch dict with tensors stacked and metadata as list.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    # Keep metadata as list (not tensorizable)
    metadata = [item["metadata"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "metadata": metadata,
        "sample_ids": sample_ids
    }


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    tokenizer: Optional[SymbolicOntologyTokenizer] = None,
    batch_size: int = 32,
    max_input_len: int = 256,
    max_target_len: int = 32,
    training_mode: str = "next_token",
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and optional validation dataloaders.
    
    Args:
        train_path: Path to training JSONL
        val_path: Optional path to validation JSONL
        tokenizer: Tokenizer instance (creates new one if None)
        batch_size: Batch size
        max_input_len: Max input sequence length
        max_target_len: Max target sequence length
        training_mode: "next_token" or "seq2seq"
        num_workers: Number of dataloader workers
        
    Returns:
        (train_loader, val_loader) tuple
    """
    if tokenizer is None:
        tokenizer = SymbolicOntologyTokenizer()
    
    train_dataset = SymbolicOntologyDataset(
        train_path, tokenizer, max_input_len, max_target_len, training_mode
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_path is not None:
        val_dataset = SymbolicOntologyDataset(
            val_path, tokenizer, max_input_len, max_target_len, training_mode
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


def test_dataset():
    """Test the dataset with a sample file."""
    from pathlib import Path
    
    # Check if we have a test file
    test_file = Path("/mnt/user-data/outputs/symbolic_inductive_train.jsonl")
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Please run generate_symbolic_ontology.py first to create the dataset.")
        return
    
    print("Testing SymbolicOntologyDataset...")
    print("=" * 50)
    
    tokenizer = SymbolicOntologyTokenizer()
    dataset = SymbolicOntologyDataset(
        str(test_file),
        tokenizer,
        max_input_len=256,
        max_target_len=32,
        training_mode="next_token"
    )
    
    # Get stats
    stats = dataset.get_metadata_stats()
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Task types: {stats['task_types']}")
    print(f"  Parsimony tests: {stats['parsimony_tests']} ({stats['parsimony_test_ratio']:.1%})")
    print(f"  Depth distribution: {stats['depth_distribution']}")
    print(f"  Tree depth distribution: {stats['tree_depth_distribution']}")
    
    # Get a sample
    print("\n" + "=" * 50)
    print("Sample 0:")
    sample = dataset[0]
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Metadata: {sample['metadata']}")
    
    # Decode sample
    input_ids = sample['input_ids'].tolist()
    # Find where padding starts
    try:
        pad_start = input_ids.index(tokenizer.pad_token_id)
    except ValueError:
        pad_start = len(input_ids)
    
    print(f"\n  Decoded input (first {min(50, pad_start)} tokens):")
    decoded = tokenizer.decode(input_ids[:min(50, pad_start)])
    print(f"    {decoded[:200]}...")
    
    # Test dataloader
    print("\n" + "=" * 50)
    print("Testing DataLoader...")
    train_loader, _ = create_dataloaders(
        str(test_file),
        tokenizer=tokenizer,
        batch_size=4,
        max_input_len=256,
        max_target_len=32
    )
    
    batch = next(iter(train_loader))
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch labels shape: {batch['labels'].shape}")
    print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  Number of metadata entries: {len(batch['metadata'])}")
    
    print("\nDataset test complete!")


if __name__ == "__main__":
    test_dataset()