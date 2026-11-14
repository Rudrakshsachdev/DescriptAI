# DESCЯA — Practical Deep Learning Exercises
## 9 Hands-On PyTorch & Transformer Exercises for Team Skill Building

**Project:** Descra — AI-Powered Product Description Generator  
**Phase:** Phase 1 — Deep Learning Foundation & Model Development  
**Academic Team:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)  
**Objective:** Hands-on code implementations reinforcing the theoretical concepts from `docs/LEARNING_ROADMAP.md`.

---

## Exercise 1: PyTorch Tensors, Memory Layout & CUDA Device Placement

### Objective
Understand tensor creation, device movement (`CPU` $\to$ `CUDA`), tensor memory contiguity, and execute a matrix multiplication on the NVIDIA RTX 4050 GPU.

### Implementation
```python
import torch

def exercise_1_solution():
    # 1. Verify CUDA device
    assert torch.cuda.is_available(), "CUDA GPU is not available!"
    device = torch.device("cuda:0")
    print(f"Active Device: {torch.cuda.get_device_name(device)}")

    # 2. Create tensors on CPU and move to CUDA
    x = torch.randn(1024, 512, device=device)
    w = torch.randn(512, 2048, device=device)

    # 3. Check memory layout and contiguity
    print(f"x shape: {x.shape}, is_contiguous: {x.is_contiguous()}")

    # 4. Perform matrix multiplication (GEMM)
    # y = x @ w -> Shape: (1024, 2048)
    y = torch.matmul(x, w)
    print(f"Result y shape: {y.shape}, device: {y.device}")

    # 5. Measure allocated VRAM
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Allocated VRAM: {allocated_mb:.2f} MB")

    # Clean up
    del x, w, y
    torch.cuda.empty_cache()
    print("Exercise 1 passed successfully!\n")

if __name__ == "__main__":
    exercise_1_solution()
```

### Explanation
- `torch.randn(..., device=device)` creates tensors directly in GPU VRAM, bypassing CPU RAM allocation.
- `torch.matmul` invokes NVIDIA cuBLAS tensor core kernels for fast floating-point matrix multiplication.
- `torch.cuda.empty_cache()` releases cached unallocated memory back to the operating system.

---

## Exercise 2: Subword Tokenization & Reconstruction with SentencePiece

### Objective
Inspect how the FLAN-T5 SentencePiece tokenizer decomposes technical product terms and reconstructs original text from token IDs.

### Implementation
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

def exercise_2_solution():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    sample_text = "Cetaphil Daily Hydrating Lotion with Hyaluronic Acid (88ml)"

    # 1. Tokenize into IDs
    token_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f"Input: {sample_text}")
    print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")

    # 2. Inspect individual subword pieces
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print("\nSubword Tokens Breakdown:")
    for tid, tok in zip(token_ids, tokens):
        print(f"  ID {tid:>5} -> {tok}")

    # 3. Decode back to string
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"\nDecoded text: '{decoded}'")
    assert decoded == sample_text, "Decoded text does not match original!"
    print("Exercise 2 passed successfully!\n")

if __name__ == "__main__":
    exercise_2_solution()
```

### Explanation
- The leading unicode character ` ` (U+2581) denotes whitespace preceding the token piece.
- Unseen technical words like "Cetaphil" are cleanly split into morphemes (`_Ce`, `t`, `aph`, `il`) rather than causing `<unk>` errors.

---

## Exercise 3: Implementing Scaled Dot-Product Self-Attention from Scratch

### Objective
Implement the fundamental attention equation $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$ in pure PyTorch.

### Implementation
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Q, K, V shapes: (batch_size, seq_len, d_k)
        # 1. Compute pairwise dot product: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. Apply optional attention mask (e.g. padding mask or causal mask)
        if mask is not None:
            # Mask positions with -1e9 so softmax produces 0 probability
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Apply softmax across keys dimension
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Multiply by Values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

def exercise_3_solution():
    batch_size, seq_len, d_k = 2, 4, 64
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    # Padding mask: batch 0 has 4 tokens, batch 1 has 3 tokens (1 pad token)
    mask = torch.tensor([
        [[1, 1, 1, 1]],
        [[1, 1, 1, 0]]
    ])  # Shape: (2, 1, 4)

    attn = ScaledDotProductAttention(d_k)
    out, weights = attn(Q, K, V, mask)

    print(f"Output shape: {out.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Batch 1 weights for last padded position (should be 0.0):")
    print(weights[1, :, -1])
    assert torch.allclose(weights[1, :, -1], torch.zeros(seq_len), atol=1e-4)
    print("Exercise 3 passed successfully!\n")

if __name__ == "__main__":
    exercise_3_solution()
```

---

## Exercise 4: Positional Encodings: Sinusoidal vs Relative Bias

### Objective
Contrast absolute sinusoidal positional encodings with relative distance matrix computation.

### Implementation
```python
import math
import torch

def get_sinusoidal_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def get_relative_distance_matrix(seq_len: int) -> torch.Tensor:
    # Generates matrix where entry (i, j) is relative offset i - j
    pos = torch.arange(seq_len)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    return rel_pos

def exercise_4_solution():
    seq_len, d_model = 6, 16
    sin_pe = get_sinusoidal_embeddings(seq_len, d_model)
    rel_mat = get_relative_distance_matrix(seq_len)

    print(f"Sinusoidal PE shape: {sin_pe.shape}")
    print(f"Relative Distance Matrix ({seq_len}x{seq_len}):")
    print(rel_mat)

    # Verify that diagonal is 0 (distance to self is 0)
    assert torch.all(torch.diag(rel_mat) == 0)
    print("Exercise 4 passed successfully!\n")

if __name__ == "__main__":
    exercise_4_solution()
```

---

## Exercise 5: Building a PyTorch Seq2Seq DataLoader with Dynamic Padding

### Objective
Build a functional `Dataset` and `DataLoader` that applies dynamic right-padding and creates proper `-100` label masks.

### Implementation
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class ToySeq2SeqDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        input_ids = self.tokenizer.encode(inp, truncation=True, max_length=128)
        target_ids = self.tokenizer.encode(tgt, truncation=True, max_length=64)
        return {"input_ids": input_ids, "target_ids": target_ids}

def collate_seq2seq(batch, pad_token_id=0):
    max_inp = max(len(x["input_ids"]) for x in batch)
    max_tgt = max(len(x["target_ids"]) for x in batch)

    batch_inp, batch_mask, batch_labels = [], [], []

    for x in batch:
        inp = x["input_ids"]
        tgt = x["target_ids"]

        # Pad inputs
        inp_padded = inp + [pad_token_id] * (max_inp - len(inp))
        mask = [1] * len(inp) + [0] * (max_inp - len(inp))

        # Pad targets and mask padding with -100
        labels = tgt + [-100] * (max_tgt - len(tgt))

        batch_inp.append(inp_padded)
        batch_mask.append(mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.tensor(batch_inp, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long)
    }

def exercise_5_solution():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    data = [
        ("Title: Lip Balm | Features: Organic beeswax", "Soothing lip balm for dry lips."),
        ("Title: Shampoo", "Clarifying cleanser.")
    ]

    dataset = ToySeq2SeqDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_seq2seq)

    batch = next(iter(loader))
    print(f"Batch input_ids shape:      {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape:          {batch['labels'].shape}")
    print(f"Labels with -100 masking:\n{batch['labels']}")
    assert -100 in batch["labels"]
    print("Exercise 5 passed successfully!\n")

if __name__ == "__main__":
    exercise_5_solution()
```

---

## Exercise 6: Computing Cross-Entropy Loss with Ignore Index (`-100`)

### Objective
Demonstrate numerically how `nn.CrossEntropyLoss(ignore_index=-100)` ignores padding positions and prevents loss distortion.

### Implementation
```python
import torch
import torch.nn as nn

def exercise_6_solution():
    vocab_size = 10
    batch_size = 2
    seq_len = 3

    # Simulated unnormalized model logits: (batch_size, seq_len, vocab_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Flatten logits to (batch_size * seq_len, vocab_size)
    flat_logits = logits.view(-1, vocab_size)

    # Targets with padding: token 0 is real, token 1 is real, token 2 is pad (-100)
    targets = torch.tensor([
        [3, 5, -100],
        [1, -100, -100]
    ]).view(-1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(flat_logits, targets)

    print(f"Total tokens in batch: {len(targets)}")
    print(f"Active non-pad tokens: {(targets != -100).sum().item()}")
    print(f"Calculated Cross-Entropy Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss computed as NaN!"
    print("Exercise 6 passed successfully!\n")

if __name__ == "__main__":
    exercise_6_solution()
```

---

## Exercise 7: Text Decoding Comparison: Greedy vs Beam Search

### Objective
Observe the difference between Greedy Search and Beam Search decoding on sample language generation.

### Implementation
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def exercise_7_solution():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.eval()

    prompt = "Generate a product description: Cetaphil moisturizing cream for sensitive skin."
    inputs = tokenizer(prompt, return_tensors="pt")

    # 1. Greedy Search
    greedy_ids = model.generate(**inputs, max_length=50, num_beams=1, do_sample=False)
    greedy_text = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

    # 2. Beam Search (num_beams=4)
    beam_ids = model.generate(
        **inputs,
        max_length=50,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    beam_text = tokenizer.decode(beam_ids[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"\nGreedy Decoding: '{greedy_text}'")
    print(f"\nBeam Search Decoding: '{beam_text}'")
    print("Exercise 7 passed successfully!\n")

if __name__ == "__main__":
    exercise_7_solution()
```

---

## Exercise 8: Implementing ROUGE-1 and ROUGE-L from Scratch

### Objective
Implement the mathematical definitions of ROUGE-1 (unigram recall) and ROUGE-L (Longest Common Subsequence recall) in pure Python.

### Implementation
```python
from collections import Counter

def compute_rouge_1(hypothesis: str, reference: str) -> dict:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    hyp_counts = Counter(hyp_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = sum(min(count, ref_counts[w]) for w, count in hyp_counts.items())

    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_l(hypothesis: str, reference: str) -> dict:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def exercise_8_solution():
    reference = "Hydrating moisturizer formulated with hyaluronic acid for dry sensitive skin."
    hypothesis = "A hydrating moisturizer with hyaluronic acid designed for sensitive skin."

    r1 = compute_rouge_1(hypothesis, reference)
    rl = compute_rouge_l(hypothesis, reference)

    print("ROUGE-1:", {k: round(v, 4) for k, v in r1.items()})
    print("ROUGE-L:", {k: round(v, 4) for k, v in rl.items()})

    assert r1["recall"] > 0.6, "ROUGE-1 recall unexpectedly low!"
    assert rl["f1"] > 0.6, "ROUGE-L F1 unexpectedly low!"
    print("Exercise 8 passed successfully!\n")

if __name__ == "__main__":
    exercise_8_solution()
```

---

## Exercise 9: Profiling GPU VRAM Allocation with PyTorch

### Objective
Quantify exact memory consumption differences between FP32 and FP16 tensors on the NVIDIA RTX 4050 GPU.

### Implementation
```python
import torch

def exercise_9_solution():
    assert torch.cuda.is_available(), "CUDA device required for VRAM profiling!"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Baseline VRAM
    base_mem = torch.cuda.memory_allocated() / (1024 ** 2)

    # 2. Allocate 10 million parameters in FP32 (4 bytes per parameter = ~40 MB)
    fp32_tensor = torch.zeros(10_000_000, dtype=torch.float32, device="cuda")
    fp32_mem = (torch.cuda.memory_allocated() / (1024 ** 2)) - base_mem

    # Clean up
    del fp32_tensor
    torch.cuda.empty_cache()

    # 3. Allocate 10 million parameters in FP16 (2 bytes per parameter = ~20 MB)
    fp16_tensor = torch.zeros(10_000_000, dtype=torch.float16, device="cuda")
    fp16_mem = (torch.cuda.memory_allocated() / (1024 ** 2)) - base_mem

    print(f"FP32 Allocation for 10M elements: {fp32_mem:.2f} MB")
    print(f"FP16 Allocation for 10M elements: {fp16_mem:.2f} MB")
    print(f"VRAM Reduction Factor: {fp32_mem / fp16_mem:.2f}x")

    assert fp16_mem < fp32_mem * 0.6, "FP16 did not provide expected memory savings!"
    print("Exercise 9 passed successfully!\n")

if __name__ == "__main__":
    exercise_9_solution()
```

