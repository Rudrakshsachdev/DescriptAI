# QUALITATIVE GENERATION ANALYSIS & FAILURE CASEBOOK
## Descra — Phase 1: Step 7 Systematic Baseline Experimentation

**Date:** November 18, 2026
**Model:** `google/flan-t5-small` (Zero-Shot Baseline, Pretrained Weights Only)

---

## 1. Executive Summary of Failure Modes

Pretrained `google/flan-t5-small` exhibits systematic zero-shot inductive biases when prompted for product descriptions. Because FLAN-T5 was heavily instruction-tuned on extractive question answering, reading comprehension, and classification tasks, it struggles to produce multi-sentence e-commerce marketing narratives without domain fine-tuning.

### Quantitative Failure Breakdown by Strategy

| Failure Mode | Greedy | Beam-2 | Beam-4 | Sample (Temp 0.7) | Sample (Temp 1.0) |
|---|---:|---:|---:|---:|---:|
| **Output Collapse (1-2 words)** | 2 | 1 | 1 | 5 | 5 |
| **Empty / EOS Termination** | 4 | 0 | 0 | 1 | 0 |
| **Prompt Echoing / Header Leak** | 48 | 57 | 68 | 24 | 17 |
| **Degenerative Repetition** | 11 | 9 | 8 | 8 | 7 |
| **Hallucinated Numbers/Units** | 5 | 5 | 2 | 10 | 12 |
| **Missing Critical Features** | 7 | 5 | 2 | 13 | 24 |
| **Generic Conversational Filler** | 1 | 0 | 0 | 0 | 0 |

---

## 2. Representative Qualitative Case Studies

### Case Study 1: Product `B07B2N36S5`

**Product Name:** Jade Roller for face - 100% Real Natural Jade Stone roller, Anti-Aging jade Beauty Skincare Tool to Rejuvenate Facial Skin Cooling, Slimming & Firming by Kemzzone

**Ground Truth Human Reference:**
> *"Royal Jade Roller The Royal Jade Roller by kemzzone is ancient Chinese beauty tool for face and neck massage that's been used in China for thousands of years and sometimes called Jade Derma Roller. It was said that it was used in Chinese medicine to aid the body's lymphatic and filtration system to ..."*

**Generated Outputs by Strategy:**

- **Greedy Search:**
  - Output: `"Kemzzone"`
  - Latency: 130.8 ms | Tok/s: 45.9
  - ROUGE-1: 0.0116 | ROUGE-2: 0.0000 | BLEU: 0.00
  - Flags: `OUTPUT_COLLAPSE`

- **Beam Search (B=2):**
  - Output: `"Kemzzone"`
  - Latency: 352.8 ms | Tok/s: 17.0
  - ROUGE-1: 0.0116 | ROUGE-2: 0.0000 | BLEU: 0.00
  - Flags: `OUTPUT_COLLAPSE`

- **Beam Search (B=4):**
  - Output: `"Kemzzone"`
  - Latency: 443.2 ms | Tok/s: 13.5
  - ROUGE-1: 0.0116 | ROUGE-2: 0.0000 | BLEU: 0.00
  - Flags: `OUTPUT_COLLAPSE`

- **Nucleus Sampling (T=0.7):**
  - Output: `"Kemzzone"`
  - Latency: 306.0 ms | Tok/s: 19.6
  - ROUGE-1: 0.0116 | ROUGE-2: 0.0000 | BLEU: 0.00
  - Flags: `OUTPUT_COLLAPSE`

- **Creative Sampling (T=1.0):**
  - Output: `"Kemzzone"`
  - Latency: 188.2 ms | Tok/s: 31.9
  - ROUGE-1: 0.0116 | ROUGE-2: 0.0000 | BLEU: 0.00
  - Flags: `OUTPUT_COLLAPSE`

### Case Study 2: Product `B09N9VH2WW`

**Product Name:** Popdigito Pet Self-Cleaning Brush for Short or Long Hair, Remove Loose or Tangled Hair for Cat and Dog, Gently Comb Undercoat and Massage Skin, Grooming and Deshedding Tool

**Ground Truth Human Reference:**
> *"Dog&Cat Pet Self-Cleaning Slicker Brush for Short and Long Hair, Grooming Dematting Tool Easily Removes Undercoat, Shedding Mats, Tangles Hair, Dander, Dirt, Loose Fur, Massages Particle, Improves Circulation Pet Supplies..."*

**Generated Outputs by Strategy:**

- **Greedy Search:**
  - Output: `"Popdigito Pet Self-Cleaning Brush for Short or Long Hair, Remove Loose or Tangled Hair for Cat and Dog, Gently Comb Undercoat and Massage Skin, Grooming and Deshedding Tool Features: - Strong function: This grooming brush can remove dead hair and dirt from pets’ fur. It works out tangles and knots, keeping pets’ coat smooth and shiny. This tool is suitable for small and medium size cats and dogs with different coat types. -"`
  - Latency: 2101.5 ms | Tok/s: 53.8
  - ROUGE-1: 0.4231 | ROUGE-2: 0.1176 | BLEU: 4.06
  - Flags: `PROMPT_ECHO`

- **Beam Search (B=2):**
  - Output: `"Popdigito Pet Self-Cleaning Brush for Short or Long Hair, Remove Loose or Tangled Hair for Cat and Dog, Gently Comb Undercoat and Massage Skin, Grooming and Deshedding Tool Features: - Strong function: This grooming brush can remove dead hair and dirt from pets’ fur. It works out tangles and knots, keeping pets’ coat smooth and shiny. This tool is suitable for small and medium size cats and dogs with different coat types."`
  - Latency: 2794.4 ms | Tok/s: 39.7
  - ROUGE-1: 0.4231 | ROUGE-2: 0.1176 | BLEU: 4.11
  - Flags: `PROMPT_ECHO`

- **Beam Search (B=4):**
  - Output: `"Popdigito Pet Self-Cleaning Brush for Short or Long Hair, Remove Loose or Tangled Hair for Cat and Dog, Gently Comb Undercoat and Massage Skin, Grooming and Deshedding Tool Features: - Strong function: This grooming brush can remove dead hair and dirt from pets’ fur. It works out tangles and knots, keeping pets’ coat smooth and shiny. This tool is suitable for small and medium size cats and dogs with different coat types."`
  - Latency: 2810.2 ms | Tok/s: 39.5
  - ROUGE-1: 0.4231 | ROUGE-2: 0.1176 | BLEU: 4.11
  - Flags: `PROMPT_ECHO`

- **Nucleus Sampling (T=0.7):**
  - Output: `"It is easy to use and easy to clean."`
  - Latency: 267.4 ms | Tok/s: 44.9
  - ROUGE-1: 0.0976 | ROUGE-2: 0.0000 | BLEU: 0.21
  - Flags: `None`

- **Creative Sampling (T=1.0):**
  - Output: `"Do not use products from Popdigito Pet Self-Cleaning Brush for short or long hair, remove hair or dirt from your fur."`
  - Latency: 761.5 ms | Tok/s: 42.0
  - ROUGE-1: 0.4444 | ROUGE-2: 0.1923 | BLEU: 2.35
  - Flags: `None`

### Case Study 3: Product `B00JNRWMYI`

**Product Name:** Samantha Brown 6-piece Travel Survival Kit - BURGUNDY

**Ground Truth Human Reference:**
> *"Samantha Brown 6-piece Travel Survival Kit Never fret about storing laundry and other important items when you travel. This smart kit features bags to keep you organized while on the go. What a great way to have a place for everything and everything in its place...."*

**Generated Outputs by Strategy:**

- **Greedy Search:**
  - Output: `"The kit is a simple, easy to use, and easy to carry."`
  - Latency: 344.4 ms | Tok/s: 52.3
  - ROUGE-1: 0.2034 | ROUGE-2: 0.0000 | BLEU: 0.37
  - Flags: `MISSING_FEATURES, GENERIC_FILLER`

- **Beam Search (B=2):**
  - Output: `"The Samantha Brown 6-piece Travel Survival Kit is a great way to keep your hands warm and dry. It is made of a sturdy material that is easy to clean and easy to use."`
  - Latency: 1147.3 ms | Tok/s: 35.7
  - ROUGE-1: 0.4146 | ROUGE-2: 0.2500 | BLEU: 19.48
  - Flags: `MISSING_FEATURES`

- **Beam Search (B=4):**
  - Output: `"BURGUNDY Features: - Crossbody bag - Clear pouch - Double-sided laundry cube"`
  - Latency: 652.6 ms | Tok/s: 39.8
  - ROUGE-1: 0.1053 | ROUGE-2: 0.0000 | BLEU: 0.20
  - Flags: `PROMPT_ECHO`

- **Nucleus Sampling (T=0.7):**
  - Output: `"Samantha Brown 6 piece Travel Survival Kit - BURGUNDY Features: - Crossbody bag - Clear pouch - Double-sided laundry cube - Removable mesh bag ."`
  - Latency: 959.9 ms | Tok/s: 45.8
  - ROUGE-1: 0.2985 | ROUGE-2: 0.1846 | BLEU: 5.47
  - Flags: `PROMPT_ECHO, UNSUPPORTED_NUMBERS`

- **Creative Sampling (T=1.0):**
  - Output: `"An easy way to safely and safely dry your backpack from the front of the backpack. Your bag contains the following materials: - Crossbody bag - Clear pouch - Single-sided laundry bag . - Double-sided duvet cover - Disinfect mesh bag ..."`
  - Latency: 1362.0 ms | Tok/s: 45.5
  - ROUGE-1: 0.1429 | ROUGE-2: 0.0244 | BLEU: 2.08
  - Flags: `None`

### Case Study 4: Product `B074XQP3B5`

**Product Name:** Nivea Men Fresh Ocean Aluminium Free deodorant roll-on for long lasting freshness with aqua fresh scent 50 ml / 1.7 fl oz

**Ground Truth Human Reference:**
> *"Nivea Men Fresh Ocean Aluminium Free deodorant roll-on for long lasting freshness with aqua fresh scent 50 ml / 1.7 fl oz..."*

**Generated Outputs by Strategy:**

- **Greedy Search:**
  - Output: `"Nivea Men Fresh Ocean Aluminium Free Deodorant Roll-on for long lasting freshness with aqua fresh scent 50 ml / 1.7 fl oz"`
  - Latency: 762.6 ms | Tok/s: 51.1
  - ROUGE-1: 1.0000 | ROUGE-2: 1.0000 | BLEU: 82.32
  - Flags: `None`

- **Beam Search (B=2):**
  - Output: `"Nivea Men Fresh Ocean Aluminium Free deodorant roll-on for long lasting freshness with aqua fresh scent 50 ml / 1.7 fl oz"`
  - Latency: 1074.7 ms | Tok/s: 36.3
  - ROUGE-1: 1.0000 | ROUGE-2: 1.0000 | BLEU: 100.00
  - Flags: `None`

- **Beam Search (B=4):**
  - Output: `"Nivea Men Fresh Ocean Aluminium Free deodorant roll-on for long lasting freshness with aqua fresh scent 50 ml / 1.7 fl oz"`
  - Latency: 892.0 ms | Tok/s: 43.7
  - ROUGE-1: 1.0000 | ROUGE-2: 1.0000 | BLEU: 100.00
  - Flags: `None`

- **Nucleus Sampling (T=0.7):**
  - Output: `"The nivea Men Fresh Ocean is a non-abrasive, non-oil-based, nonrenewable, and non-hazardous product. It is manufactured by Nivea and made by the company Beiersdorf."`
  - Latency: 1353.0 ms | Tok/s: 41.4
  - ROUGE-1: 0.1569 | ROUGE-2: 0.1224 | BLEU: 5.15
  - Flags: `None`

- **Creative Sampling (T=1.0):**
  - Output: `"Nivea Men Fresh Ocean Aluminium Free Deodorant roll-on for long lasting freshness with aqua fresh scent 50 ml / 1.7 fl oz Features: - Nive A Men Fresh Orange Aluminium Free deodorant rolls on for long term freshness With aqua fresh smell 50 - weight : 1 unit - unit count: 1.7 Fl Oz"`
  - Latency: 2058.1 ms | Tok/s: 39.8
  - ROUGE-1: 0.6133 | ROUGE-2: 0.6027 | BLEU: 34.07
  - Flags: `PROMPT_ECHO, REPETITIVE_NGRAMS`

### Case Study 5: Product `B08W53WBGB`

**Product Name:** Zhaocaimao 20PCS Sample Containers 5ml Plastic Small Bottle Test Tube Mini Bottles Storage Containers White

**Ground Truth Human Reference:**
> *"Compact size. Good sealing performance. Height: approx.42mm Convenient to use and carry. Capacity: 5ml Package included: 20pcs x sample containers Quantity: 20pcs Material: Polyethylene plastic (PE) Feature: Color: translucent Easily screws together for compact storage of small pills, powders, capsu..."*

**Generated Outputs by Strategy:**

- **Greedy Search:**
  - Output: `""`
  - Latency: 120.2 ms | Tok/s: 49.9
  - ROUGE-1: 0.0000 | ROUGE-2: 0.0000 | BLEU: 0.00
  - Flags: `EMPTY_OUTPUT`

- **Beam Search (B=2):**
  - Output: `"The product is made of polyethylene plastic (PE) - color is translucent - Compact size. Good sealing performance. - Easily screws together for compact storage of small pills, powders, capsules and other solid."`
  - Latency: 1012.6 ms | Tok/s: 48.4
  - ROUGE-1: 0.6154 | ROUGE-2: 0.5000 | BLEU: 31.83
  - Flags: `None`

- **Beam Search (B=4):**
  - Output: `"Product Description: Zhaocaimao 20PCS Sample Containers 5ml Plastic Small Bottle Test Tube Mini Bottles Storage Containers White Features: - Material: Polyethylene plastic (PE) - Color: translucent - Compact size. Good sealing performance. - Easily screws together for compact storage of small pills, powders, capsules and other solid."`
  - Latency: 2101.9 ms | Tok/s: 37.6
  - ROUGE-1: 0.6593 | ROUGE-2: 0.4944 | BLEU: 43.53
  - Flags: `PROMPT_ECHO`

- **Nucleus Sampling (T=0.7):**
  - Output: `"- Suitable for use in the laboratory, for a reusable container."`
  - Latency: 598.2 ms | Tok/s: 31.8
  - ROUGE-1: 0.1034 | ROUGE-2: 0.0000 | BLEU: 0.07
  - Flags: `MISSING_FEATURES`

- **Creative Sampling (T=1.0):**
  - Output: `"Zhaocaimao 20PCS Sample Containers 5ml Plastic Small Bottle Test Tube Mini Bottles Storage Containers White Features: - Material: Polyethylene plastic (PE) - Color: translucent - Compact size. Good sealing performance. - Easily screws together for compact storage of small pills, powders, capsules and other solid. /small. jpg"`
  - Latency: 1827.4 ms | Tok/s: 45.4
  - ROUGE-1: 0.6593 | ROUGE-2: 0.4944 | BLEU: 43.61
  - Flags: `PROMPT_ECHO`

## 3. Detailed Failure Mode Analysis

### 3.1 Output Collapse into Named Entities
- **Observed Behavior:** The model outputs only a brand name or a product title fragment (e.g. `"Kemzzone"` or `"Jade Roller"`).
- **Mechanism:** The cross-attention mechanism in the encoder-decoder focuses strongly on high-salience capitalized tokens in the input prompt. Without supervised target length conditioning, the decoder's EOS probability spikes after generating the salient named entity.

### 3.2 Prompt Echoing and Structural Mimicry
- **Observed Behavior:** Output reproduces the prompt's `"Features: - "` bullet structure.
- **Mechanism:** In greedy search, the model copies n-grams directly from the encoder representations due to high self-attention weights on prompt headers.

### 3.3 Stochastic Sampling Divergence
- **Observed Behavior:** Temperature 1.0 occasionally outputs hallucinated ingredients or ungrounded quantities.
- **Mechanism:** Flattening the softmax distribution ($P(w_i) \propto \exp(z_i / T)$) allows low-probability tail tokens from unrelated domains into the candidate pool.
