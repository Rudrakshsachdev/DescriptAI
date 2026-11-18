# FAILURE ANALYSIS & HEURISTIC ERROR TAXONOMY
## Descra — Phase 1: Step 9 Held-Out Test Split (N=480 Records)

### 1. Automated Detection Methodology

Every generated prediction across the 2,880 evaluation passes was processed through an automated multi-criteria heuristic classifier (`src/descra/evaluation/failure_analysis.py`). The taxonomy evaluates 6 distinct structural and semantic error patterns:

1. **EMPTY_OUTPUT:** Generation is blank or contains only whitespace.
2. **BRAND_OR_EXTREMELY_SHORT:** Output is $\le 3$ words, collapsing into a bare brand name or truncated token.
3. **PROMPT_ECHO:** Output reproduces instruction or prompt template syntax (e.g., 'Generate a clear and engaging...', 'Features:').
4. **REPETITIVE_OUTPUT:** Output contains $\ge 2$ identical n-gram repetitions ($n=3$).
5. **UNSUPPORTED_NUMBERS:** Output introduces specific numerical quantities, percentages, or units not present in the input metadata.
6. **GENERIC_FILLER:** Output is purely generic conversational filler with fewer than 25 words.

### 2. Comprehensive Failure Distribution Table Across All Strategies

| Strategy | Model | Total Evaluated | Total Failures | Failure Rate (%) | Prompt Echo | Repetitive | Unsupp. Numbers | Brand Collapse | Empty |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| greedy | Baseline | 480 | 303 | **63.12%** | 238 | 152 | 16 | 1 | 12 |
| greedy | Finetuned | 480 | 59 | **12.29%** | 1 | 55 | 2 | 1 | 0 |
| beam_search_2 | Baseline | 480 | 262 | **54.58%** | 249 | 24 | 26 | 1 | 2 |
| beam_search_2 | Finetuned | 480 | 15 | **3.12%** | 3 | 6 | 7 | 0 | 0 |
| beam_search_4 | Baseline | 480 | 275 | **57.29%** | 270 | 16 | 16 | 1 | 0 |
| beam_search_4 | Finetuned | 480 | 11 | **2.29%** | 3 | 4 | 4 | 0 | 0 |

### 3. Key Failure Analysis Insights

1. **Near-Total Elimination of Prompt Echoing:** Pretrained FLAN-T5 suffered from prompt echoing in over 50% of test examples (e.g. 270 cases in Beam-4). Supervised fine-tuning taught the model to immediately start generating description content, reducing prompt echoing to **3 cases (0.6%)**.
2. **Unsupported Numbers Drop by 75%:** In Beam-4, baseline FLAN-T5 hallucinated unsupported numbers in 16 test examples (3.3%). Fine-tuning reduced this to **4 examples (0.8%)**.
3. **Repetition Mitigation:** Fine-tuned beam search with `no_repeat_ngram_size=3` suppressed repeated n-grams from 152 cases in baseline greedy down to **4 cases in fine-tuned Beam-4**.
