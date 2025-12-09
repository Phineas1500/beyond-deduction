#!/usr/bin/env python3
"""
Modal training script for INABHYD symbolic FOL reasoning tasks.

This uses the original INABHYD data generation with symbolic FOL rendering,
which correctly prevents the "unique property shortcut" that plagued the
isomorphic module.

Usage:
    modal run train_inabhyd.py --hops 3 --task property
    modal run train_inabhyd.py --hops 4 --task property
    modal run train_inabhyd.py --hops 5 --task property
"""
import modal
from pathlib import Path

app = modal.App("inabhyd-symbolic-training")

# Modal image with dependencies and local INABHYD module
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy")
    .add_local_dir(
        Path(__file__).parent / "inabhyd",
        remote_path="/root/inabhyd"
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
)
def train(
    hops: int = 3,
    task: str = "property",
    difficulty: str = "hard",
    mix_hops: bool = True,
    num_train: int = 2000,
    num_val: int = 400,
    n_layers: int = 4,
    d_model: int = 64,
    d_ff: int = 256,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
    eval_every: int = 50,
    seed: int = 42,
    early_stop_patience: int = 10,
):
    import sys
    sys.path.insert(0, "/root")

    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Import INABHYD symbolic generator
    from inabhyd.symbolic import generate_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Config
    config = {
        "hops": hops,
        "task": task,
        "difficulty": difficulty,
        "mix_hops": mix_hops,
        "num_train": num_train,
        "num_val": num_val,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
    }
    print(f"Config: {config}")

    # =========================================================================
    # TOKENIZER
    # =========================================================================
    class SymbolicTokenizer:
        """Simple tokenizer for symbolic FOL."""

        def __init__(self, max_seq_len=544):
            self.max_seq_len = max_seq_len
            self._build_vocab()

        def _build_vocab(self):
            self.token_to_id = {}
            self.id_to_token = {}
            idx = 0

            # Special tokens
            for tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            # Structure tokens
            for tok in ["[WORLD_MODEL]", "[OBSERVATIONS]", "[TASK]", "[ANSWER]"]:
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            # FOL tokens
            for tok in ["forall", "x", ":", "->", "(", ")", "\n"]:
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            # Concepts c0-c49
            for i in range(50):
                tok = f"c{i}"
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            # Properties p1-p20 and ~p1-~p20
            for i in range(1, 21):
                tok = f"p{i}"
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1
                tok = f"~p{i}"
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            # Entities e0-e49
            for i in range(50):
                tok = f"e{i}"
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            # Task description tokens
            for tok in ["Infer", "the", "hidden", "axiom", "(s)."]:
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

            self.vocab_size = idx
            self.pad_token_id = self.token_to_id["<PAD>"]
            self.bos_token_id = self.token_to_id["<BOS>"]
            self.eos_token_id = self.token_to_id["<EOS>"]
            self.unk_token_id = self.token_to_id["<UNK>"]

        def tokenize(self, text: str) -> list:
            """Tokenize text into tokens."""
            tokens = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Handle special markers
                if line.startswith("["):
                    tokens.append(line)
                    tokens.append("\n")
                    continue

                # Handle task description
                if line.startswith("Infer"):
                    tokens.extend(["Infer", "the", "hidden", "axiom", "(s)."])
                    tokens.append("\n")
                    continue

                # Handle FOL statements
                # "forall x: c1(x) -> c0(x)" or "c0(e0)" or "p1(e0)"
                if line.startswith("forall"):
                    # forall x: c1(x) -> c0(x)
                    tokens.extend(["forall", "x", ":"])
                    rest = line[len("forall x:"):].strip()
                    parts = rest.split("->")
                    # Left side: c1(x)
                    left = parts[0].strip()
                    concept = left.replace("(x)", "")
                    tokens.extend([concept, "(", "x", ")", "->"])
                    # Right side: c0(x) or p1(x) or ~p1(x)
                    right = parts[1].strip()
                    pred = right.replace("(x)", "")
                    tokens.extend([pred, "(", "x", ")"])
                else:
                    # Ground fact: c0(e0) or p1(e0) or ~p1(e0)
                    # Extract predicate and entity
                    pred = line.split("(")[0]
                    entity = line.split("(")[1].replace(")", "")
                    tokens.extend([pred, "(", entity, ")"])

                tokens.append("\n")

            return tokens

        def encode(self, text: str, add_special_tokens: bool = True) -> list:
            """Encode text to token IDs."""
            tokens = self.tokenize(text)
            ids = []
            if add_special_tokens:
                ids.append(self.bos_token_id)
            for tok in tokens:
                ids.append(self.token_to_id.get(tok, self.unk_token_id))
            if add_special_tokens:
                ids.append(self.eos_token_id)
            return ids

        def decode(self, ids: list, skip_special: bool = True) -> str:
            """Decode token IDs to text."""
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            tokens = []
            for i in ids:
                if skip_special and i in special:
                    continue
                tokens.append(self.id_to_token.get(i, "<UNK>"))
            return "".join(tokens)

    tokenizer = SymbolicTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # =========================================================================
    # DATA GENERATION
    # =========================================================================
    # If hops=0, train on mixed depths (2,3,4,5)
    if hops == 0:
        print(f"\nGenerating {num_train} training samples (MIXED hops 2-5, task={task}, mix_hops={mix_hops})...")
        train_data = []
        samples_per_hop = num_train // 4
        for h in [2, 3, 4, 5]:
            data = generate_dataset(
                num_samples=samples_per_hop,
                hops=h,
                task_type=task,
                difficulty=difficulty,
                mix_hops=mix_hops,
                seed=seed + h * 1000,
            )
            train_data.extend(data)
            print(f"  hops={h}: {len(data)} samples")
        import random
        random.shuffle(train_data)
        print(f"Generated {len(train_data)} total training samples")

        print(f"Generating {num_val} validation samples (MIXED)...")
        val_data = []
        val_per_hop = num_val // 4
        for h in [2, 3, 4, 5]:
            data = generate_dataset(
                num_samples=val_per_hop,
                hops=h,
                task_type=task,
                difficulty=difficulty,
                mix_hops=mix_hops,
                seed=seed + 100000 + h * 1000,
            )
            val_data.extend(data)
        random.shuffle(val_data)
        print(f"Generated {len(val_data)} total validation samples")
    else:
        print(f"\nGenerating {num_train} training samples (hops={hops}, task={task}, mix_hops={mix_hops})...")
        train_data = generate_dataset(
            num_samples=num_train,
            hops=hops,
            task_type=task,
            difficulty=difficulty,
            mix_hops=mix_hops,
            seed=seed,
        )
        print(f"Generated {len(train_data)} training samples")

        print(f"Generating {num_val} validation samples...")
        val_data = generate_dataset(
            num_samples=num_val,
            hops=hops,
            task_type=task,
            difficulty=difficulty,
            mix_hops=mix_hops,
            seed=seed + 100000,
        )
        print(f"Generated {len(val_data)} validation samples")

    # Show sample
    print(f"\n[Sample Input]:\n{train_data[0]['input']}")
    print(f"\n[Sample Target]: {train_data[0]['target']}")

    # =========================================================================
    # DATASET
    # =========================================================================
    class ReasoningDataset(Dataset):
        def __init__(self, samples, tokenizer, max_input=512, max_target=32):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_input = max_input
            self.max_target = max_target

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # Encode
            input_ids = self.tokenizer.encode(sample["input"], add_special_tokens=True)
            target_ids = self.tokenizer.encode(sample["target"], add_special_tokens=False)
            target_ids.append(self.tokenizer.eos_token_id)

            # Truncate
            if len(input_ids) > self.max_input:
                input_ids = input_ids[:self.max_input]
            if len(target_ids) > self.max_target:
                target_ids = target_ids[:self.max_target]

            # Remove trailing EOS from input if present
            if input_ids and input_ids[-1] == self.tokenizer.eos_token_id:
                input_ids = input_ids[:-1]

            # Concatenate: input + [ANSWER] + \n + target
            answer_id = self.tokenizer.token_to_id["[ANSWER]"]
            newline_id = self.tokenizer.token_to_id["\n"]
            full_ids = input_ids + [answer_id, newline_id] + target_ids

            # Labels: -100 for input, actual IDs for target
            labels = [-100] * (len(input_ids) + 2) + target_ids

            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=tokenizer.pad_token_id))
            labels.append(F.pad(item["labels"], (0, pad_len), value=-100))
            attention_mask.append(torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ]))

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }

    train_dataset = ReasoningDataset(train_data, tokenizer, max_input=3500, max_target=500)
    val_dataset = ReasoningDataset(val_data, tokenizer, max_input=3500, max_target=500)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # =========================================================================
    # MODEL
    # =========================================================================
    class Attention(nn.Module):
        def __init__(self, d_embed, d_model, dropout=0.0):
            super().__init__()
            self.proj_q = nn.Linear(d_embed, d_model)
            self.proj_k = nn.Linear(d_embed, d_model)
            self.proj_v = nn.Linear(d_embed, d_model)
            self.proj_out = nn.Linear(d_model, d_embed)
            self.scale = d_model ** 0.5
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
            Q, K, V = self.proj_q(x), self.proj_k(x), self.proj_v(x)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            return self.proj_out(torch.matmul(attn, V))

    class TransformerBlock(nn.Module):
        def __init__(self, d_embed, d_model, d_ff, dropout=0.0):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_embed)
            self.attn = Attention(d_embed, d_model, dropout)
            self.ln2 = nn.LayerNorm(d_embed)
            self.ff = nn.Sequential(
                nn.Linear(d_embed, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_embed),
                nn.Dropout(dropout),
            )

        def forward(self, x, mask):
            x = x + self.attn(self.ln1(x), mask)
            x = x + self.ff(self.ln2(x))
            return x

    class Transformer(nn.Module):
        def __init__(self, vocab_size, d_model, d_ff, n_layers, max_seq_len, pad_id=0):
            super().__init__()
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            self.d_embed = d_model + max_seq_len  # Concatenated pos embedding

            self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
            self.register_buffer('pos_emb', torch.eye(max_seq_len))

            self.layers = nn.ModuleList([
                TransformerBlock(self.d_embed, d_model, d_ff)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(self.d_embed)
            self.head = nn.Linear(self.d_embed, vocab_size, bias=False)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)

        def forward(self, input_ids):
            B, T = input_ids.shape
            tok = self.tok_emb(input_ids)
            pos = self.pos_emb[:T, :].unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([tok, pos], dim=-1)

            mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

            for layer in self.layers:
                x = layer(x, mask)

            return self.head(self.ln_f(x))

    max_seq_len = 4096  # Increased for deeper trees (hops=5 can have ~3000 char inputs)
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_token_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # =========================================================================
    # TRAINING
    # =========================================================================
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    def compute_accuracy(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(ids)
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                for i in range(ids.shape[0]):
                    m = mask[i]
                    if m.any():
                        if torch.equal(preds[i][m], labels[i][m]):
                            correct += 1
                    total += 1
        model.train()
        return correct / total if total > 0 else 0.0

    print(f"\nTraining for {epochs} epochs ({total_steps} steps)...")
    print(f"Hops: {hops} (requires {hops - 1} reasoning steps)")
    print(f"Early stopping patience: {early_stop_patience} evals without improvement")
    print("=" * 70)

    import time
    start = time.time()
    best_val = 0.0
    step = 0
    evals_without_improvement = 0
    model_save_path = f"/root/inabhyd_hops{hops}_model.pt"

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(ids)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % eval_every == 0:
                train_acc = compute_accuracy(model, train_loader)
                val_acc = compute_accuracy(model, val_loader)
                elapsed = time.time() - start
                marker = ""

                if val_acc > best_val:
                    best_val = val_acc
                    torch.save(model.state_dict(), model_save_path)
                    evals_without_improvement = 0
                    marker = " * (saved)"
                else:
                    evals_without_improvement += 1

                print(f"Step {step:5d} | Loss: {loss.item():.4f} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | {elapsed:.0f}s{marker}")

                # Early stopping
                if best_val >= 0.99 and evals_without_improvement >= early_stop_patience:
                    print(f"\nEarly stopping: {early_stop_patience} evals without improvement at {best_val*100:.1f}% accuracy")
                    break

        if best_val >= 0.99 and evals_without_improvement >= early_stop_patience:
            break

    print("=" * 70)
    print(f"Training complete. Best validation accuracy: {best_val*100:.1f}%")

    # Test generalization to different hop counts
    print("\n" + "=" * 70)
    print("GENERALIZATION TEST: Accuracy at different hop counts")
    print("=" * 70)

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    for test_hops in [2, 3, 4, 5]:
        try:
            test_data = generate_dataset(
                num_samples=200,
                hops=test_hops,
                task_type=task,
                difficulty=difficulty,
                mix_hops=mix_hops,
                seed=seed + 200000 + test_hops,
            )
            test_dataset = ReasoningDataset(test_data, tokenizer, max_input=3500, max_target=500)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_acc = compute_accuracy(model, test_loader)
            trained_marker = " (trained)" if test_hops == hops else ""
            print(f"  hops={test_hops}: {test_acc*100:.1f}%{trained_marker}")
        except Exception as e:
            print(f"  hops={test_hops}: Error - {e}")

    # Return model bytes for downloading
    with open(model_save_path, "rb") as f:
        model_bytes = f.read()

    return {
        "best_val_acc": best_val,
        "hops": hops,
        "task": task,
        "model_bytes": model_bytes,
        "model_params": n_params,
    }


@app.local_entrypoint()
def main(
    hops: int = 3,
    task: str = "property",
    difficulty: str = "hard",  # Use HARD for more variation
    mix_hops: bool = True,  # Hide axioms at multiple tree levels
    num_train: int = 2000,
    num_val: int = 400,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    result = train.remote(
        hops=hops,
        task=task,
        difficulty=difficulty,
        mix_hops=mix_hops,
        num_train=num_train,
        num_val=num_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Save model locally
    if "model_bytes" in result:
        model_path = f"inabhyd_hops{hops}_model.pt"
        with open(model_path, "wb") as f:
            f.write(result["model_bytes"])
        print(f"\nModel saved to: {model_path}")
        print(f"Model parameters: {result['model_params']:,}")

    print(f"\nBest validation accuracy: {result['best_val_acc']*100:.1f}%")
