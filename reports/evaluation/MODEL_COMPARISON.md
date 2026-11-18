# COMPREHENSIVE MODEL COMPARISON: BASELINE VS. FINE-TUNED
## Descra — Phase 1: Step 9 Final Benchmark

### 1. Architectural & Configuration Comparison

| Attribute | Pretrained Baseline (Model A) | Supervised Fine-Tuned (Model B) |
|:---|:---|:---|
| **Architecture** | FLAN-T5-small Seq2Seq | FLAN-T5-small Seq2Seq |
| **Parameter Count** | 76,961,152 parameters | 76,961,152 parameters |
| **Pretraining Objective** | Multi-task instruction tuning | Multi-task instruction tuning |
| **Domain Adaptation** | Zero-shot prompting | Supervised Fine-Tuning (3,828 Amazon records, 4 epochs) |
| **Checkpoint Path** | `google/flan-t5-small` | `checkpoints/sft/best_checkpoint` |
| **Precision Runtime** | `bfloat16` / `float16` | `bfloat16` / `float16` |

### 2. Strategy Trade-Off Analysis

1. **Greedy Decoding:**
   - Baseline: ROUGE-1 = 0.3217, Latency = 2982.9 ms, Failures = 63.1%
   - Fine-Tuned: ROUGE-1 = 0.4466, Latency = 1683.5 ms, Failures = 12.3%
   - **Assessment:** Fine-tuning transforms greedy decoding from a repetitive, echo-prone failure mode into a robust, high-throughput generator.

2. **Beam Search ($k=2$):**
   - Baseline: ROUGE-1 = 0.3921, Latency = 4406.8 ms, Failures = 54.6%
   - Fine-Tuned: ROUGE-1 = 0.4484, Latency = 1802.6 ms, Failures = 3.1%
   - **Assessment:** **Recommended production strategy.** Delivers the highest overall ROUGE-1 F1 on held-out test data (0.4484) with 97.3% heuristic consistency.

3. **Beam Search ($k=4$):**
   - Baseline: ROUGE-1 = 0.4039, Latency = 4146.9 ms, Failures = 57.3%
   - Fine-Tuned: ROUGE-1 = 0.4456, Latency = 1477.1 ms, Failures = 2.3%
   - **Assessment:** Lowest failure rate (2.29%) and highest consistency (98.5%), slightly lower lexical overlap due to concise phrasing.

### 3. Length & Output Distribution Comparison

| Model | Strategy | Mean Tokens | Median Tokens | Min Tokens | Max Tokens | Empty % | Short (<=10) % | Ceiling (>=127) % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | greedy | 90.9 | 129.0 | 3 | 129 | 2.50% | 0.83% | 54.58% |
| Baseline | beam_search_2 | 68.5 | 61.0 | 6 | 129 | 0.42% | 0.83% | 17.08% |
| Baseline | beam_search_4 | 64.5 | 59.0 | 7 | 129 | 0.00% | 0.21% | 11.46% |
| Finetuned | greedy | 57.4 | 44.0 | 5 | 129 | 0.00% | 0.42% | 16.46% |
| Finetuned | beam_search_2 | 52.7 | 40.0 | 11 | 129 | 0.00% | 0.00% | 10.00% |
| Finetuned | beam_search_4 | 50.0 | 39.0 | 11 | 129 | 0.00% | 0.00% | 7.29% |
