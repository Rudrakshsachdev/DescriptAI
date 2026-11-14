# DESCЯA — Comprehensive NLP & Deep Learning Glossary
## Essential Terminology, Mathematical Formulations, and Project Context

**Project:** Descra — AI-Powered Product Description Generator  
**Phase:** Phase 1 — Deep Learning Foundation & Model Development  
**Academic Evaluation Team:** Rudraksh Sachdeva, Shivansee, K Jahnvi, Kriti Abrol  

---

## 1. General Deep Learning & Neural Networks

### 1.1 Activation Function
- **Definition:** A non-linear mathematical operation applied element-wise to the output of a linear layer, allowing neural networks to learn non-linear patterns and complex decision boundaries.
- **Formula:** $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$
- **Descra Context:** FLAN-T5 uses **GEGLU** (Gated GELU) activations in its feed-forward sublayers to enhance gradient propagation compared to standard ReLU.

### 1.2 Backpropagation
- **Definition:** The fundamental algorithm for calculating the gradient of a scalar loss function with respect to all learnable network parameters by applying the multivariate chain rule recursively backward from output to input.
- **Formula:** $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} \cdot (\mathbf{a}^{(l-1)})^T$
- **Descra Context:** PyTorch's `loss.backward()` computes gradients across all 60 million parameters of the FLAN-T5 Small model.

### 1.3 Batch Size
- **Definition:** The number of independent training samples processed concurrently in a single forward and backward pass before updating model weights.
- **Descra Context:** In `configs/training.yaml`, physical `per_device_train_batch_size` is set to `8` to stay safely within the 6 GB VRAM capacity of the RTX 4050 GPU.

### 1.4 Computational Graph
- **Definition:** A directed acyclic graph (DAG) representing mathematical operations and variable tensors, used by frameworks like PyTorch to track operations dynamically for automatic differentiation (`autograd`).
- **Descra Context:** PyTorch constructs dynamic graphs for every mini-batch of product prompt inputs and target descriptions.

### 1.5 Gradient Clipping
- **Definition:** A regularization technique that scales down parameter gradients if their total Euclidean norm exceeds a specified threshold, preventing gradient explosions in deep networks.
- **Formula:** $\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{\text{max\_norm}}{\|\mathbf{g}\|_2}\right)$
- **Descra Context:** Set to `max_grad_norm: 1.0` in Descra's training configuration to stabilize early fine-tuning steps.

### 1.6 Layer Normalization (LayerNorm)
- **Definition:** A normalization method that normalizes activations across feature dimensions for each individual training sample independently of batch size.
- **Formula:** $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$
- **Descra Context:** FLAN-T5 replaces standard LayerNorm with **RMSNorm**, omitting the mean subtraction step ($\mu$) to reduce memory bandwidth by ~7%.

### 1.7 Learning Rate Schedule
- **Definition:** A predefined schedule that adjusts the learning rate $\eta$ during training, typically starting with a warmup phase followed by a decay phase.
- **Descra Context:** Descra utilizes a linear warmup over the first 6% of training steps followed by linear decay to 0, preventing destructive updates to pre-trained weights in early epochs.

### 1.8 Loss Function
- **Definition:** A mathematical objective measuring the discrepancy between a model's predicted output distribution and the true ground-truth targets.
- **Descra Context:** Descra uses **Categorical Cross-Entropy Loss** with `ignore_index=-100` so that `<pad>` tokens do not contribute to parameter updates.

### 1.9 Overfitting
- **Definition:** A modeling failure where a network memorizes noise and idiosyncrasies of the training data, leading to low training loss but poor generalization to unseen validation and test data.
- **Descra Context:** Monitored via validation loss in Descra; mitigated using dropout ($0.1$), weight decay ($0.01$), and early stopping with patience $3$.

### 1.10 Weight Decay
- **Definition:** An $\ell_2$ regularization penalty proportional to the squared magnitude of weights, encouraging the network to prefer smaller, distributed weights that generalize better.
- **Formula:** $\mathcal{L}_{\text{total}} = \mathcal{L}_0 + \frac{\lambda}{2} \|\mathbf{W}\|_2^2$
- **Descra Context:** Decoupled in AdamW with $\lambda = 0.01$, ensuring true weight shrinkage without distorting adaptive moment estimation.

---

## 2. Natural Language Processing & Tokenization

### 2.1 Autoregressive Language Model
- **Definition:** A model that generates text sequentially, predicting the next token conditioned on all previously generated tokens: $P(y_t \mid y_{<t}, \mathbf{x})$.
- **Descra Context:** The FLAN-T5 decoder operates autoregressively during inference, emitting one description token at a time until the `</s>` (EOS) token is produced.

### 2.2 Byte-Pair Encoding (BPE)
- **Definition:** A subword tokenization algorithm that iteratively merges the most frequent pairs of adjacent characters or character sequences in a text corpus.
- **Descra Context:** The foundational subword concept behind SentencePiece, ensuring rare words are split into meaningful morphemes rather than `<unk>`.

### 2.3 Context Window
- **Definition:** The maximum number of tokens a Transformer can accept and process simultaneously in its input sequence.
- **Descra Context:** Configured to `max_input_length: 512` in Descra, accommodating 95.8% of product feature prompts without truncation.

### 2.4 Downstream Task
- **Definition:** A specialized application or domain (e.g., e-commerce description generation) to which a pre-trained foundation model is adapted through fine-tuning.
- **Descra Context:** Adapting general FLAN-T5 instruction checkpoints to the specific task of generating engaging, factual Amazon product descriptions.

### 2.5 Dynamic Padding
- **Definition:** The practice of padding sequences in a mini-batch to the length of the longest sequence in that specific batch, rather than padding all batches to a global maximum sequence length.
- **Descra Context:** Implemented using Hugging Face's `DataCollatorForSeq2Seq` to minimize wasteful computation on `<pad>` tokens.

### 2.6 Out-Of-Vocabulary (OOV)
- **Definition:** Words or symbols encountered during inference that were not present in the tokenizer's fixed vocabulary dictionary.
- **Descra Context:** SentencePiece reduces the OOV rate to 0.0% by falling back to subword syllables or individual bytes when encountering novel brand names.

### 2.7 Pre-training
- **Definition:** Self-supervised training of a neural network on massive unannotated text corpora (e.g., C4, Wikipedia) using denoising or causal language modeling objectives.
- **Descra Context:** FLAN-T5 was pre-trained on Google's 750GB C4 dataset using a span corruption objective ($\text{<extra\_id\_0>}$).

### 2.8 SentencePiece
- **Definition:** A language-independent subword tokenizer that treats input text as a raw stream of characters (including spaces as `_`), eliminating reliance on language-specific rule-based tokenizers.
- **Descra Context:** FLAN-T5 Small uses a SentencePiece Unigram vocabulary of 32,100 tokens.

### 2.9 Sequence-to-Sequence (Seq2Seq)
- **Definition:** A neural network architecture pattern consisting of an encoder that reads an input sequence and a decoder that emits a target sequence of potentially different length.
- **Descra Context:** The core architecture of Descra: mapping structured input sequences (bullet features) to output prose descriptions.

### 2.10 Special Tokens
- **Definition:** Reserved tokens in a vocabulary that convey structural syntax or control signals rather than semantic words.
- **Descra Context:** FLAN-T5 utilizes `<pad>` (ID 0), `</s>` (EOS, ID 1), `<unk>` (ID 2), and `<extra_id_k>` (sentinel spans).

---

## 3. Transformer Architecture & Attention Mechanics

### 3.1 Attention Mask
- **Definition:** A binary or float tensor indicating which tokens should be attended to (value 1 or 0) and which tokens correspond to padding and must be ignored (value 0 or $-\infty$).
- **Descra Context:** Passed as `attention_mask` into the encoder so that `<pad>` tokens do not participate in self-attention dot products.

### 3.2 Causal Masking
- **Definition:** An upper-triangular mask of $-\infty$ values applied to the decoder self-attention matrix to prevent token $t$ from attending to future tokens $t+1, \dots, T$.
- **Formula:** $\mathbf{M}_{i, j} = -\infty \quad \forall j > i$
- **Descra Context:** Enforces autoregressive causality in the FLAN-T5 decoder during both training and inference.

### 3.3 Cross-Attention (Encoder-Decoder Attention)
- **Definition:** An attention sublayer in the decoder where queries ($\mathbf{Q}$) originate from the decoder's prior hidden states, while keys ($\mathbf{K}$) and values ($\mathbf{V}$) originate from the encoder's output representations.
- **Formula:** $\text{CrossAttention} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{dec}}\mathbf{K}_{\text{enc}}^T}{\sqrt{d_k}}\right)\mathbf{V}_{\text{enc}}$
- **Descra Context:** The primary mechanism ensuring generated marketing text is conditioned directly on the product's input features.

### 3.4 Key, Query, Value (K, Q, V)
- **Definition:** The three core projections in attention mechanisms, inspired by information retrieval: Query represents the item seeking information, Key represents the index/address of items, and Value represents the actual information content.
- **Descra Context:** In FLAN-T5 Small, each projection has dimension $d_k = d_v = 64$ across $h = 8$ heads ($512 = 8 \times 64$).

### 3.5 Multi-Head Attention (MHA)
- **Definition:** Running the scaled dot-product attention mechanism multiple times in parallel with separate learned projection matrices, allowing the model to attend to different representation subspaces simultaneously.
- **Formula:** $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$
- **Descra Context:** FLAN-T5 Small employs 8 heads in each of its 6 encoder and 6 decoder layers.

### 3.6 Relative Position Bias
- **Definition:** Adding a learnable scalar value to the pre-softmax attention logit based on the relative index offset $i - j$ between two tokens, rather than adding static positional vectors to the input embeddings.
- **Formula:** $\mathbf{A}_{i, j} = \text{softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_k}} + b_{i - j}\right)$
- **Descra Context:** T5 maps 128 relative distances into 32 logarithmic buckets, enabling smooth generalization across variable sequence lengths.

### 3.7 Scaled Dot-Product Attention
- **Definition:** The fundamental attention equation computing token compatibility via scaled matrix multiplication.
- **Formula:** $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$
- **Descra Context:** Implemented natively in PyTorch with FlashAttention/scaled_dot_product_attention kernels for maximum GPU memory efficiency.

### 3.8 Teacher Forcing
- **Definition:** A training procedure for autoregressive sequence models where ground-truth tokens from the training set are supplied as inputs to the decoder at each step, rather than feeding back the model's own predictions.
- **Descra Context:** Enables full parallelization of the 6 decoder layers during training, drastically accelerating epoch runtimes.

---

## 4. Text Generation & Decoding

### 4.1 Beam Search
- **Definition:** A heuristic search algorithm that maintains the top $B$ most probable sequence hypotheses (beams) at each decoding time step, expanding all candidates and pruning back to $B$.
- **Descra Context:** Set to `num_beams: 4` in Descra's production inference config to generate globally cohesive, high-quality product copy.

### 4.2 Greedy Decoding
- **Definition:** The simplest decoding strategy, selecting the single token with the highest predicted probability at each step: $\hat{y}_t = \arg\max P(w \mid y_{<t})$.
- **Descra Context:** Used for fast validation loss evaluations where latency is prioritized over exhaustive sequence search.

### 4.3 Length Penalty
- **Definition:** An exponent $\alpha$ applied to sequence length when scoring beam hypotheses, neutralizing beam search's mathematical preference for shorter candidate sentences.
- **Formula:** $\text{penalty} = \left(\frac{5 + |\mathbf{y}|}{6}\right)^\alpha$
- **Descra Context:** Configured to `length_penalty: 1.0` in `configs/model.yaml` to ensure generated descriptions reach complete, descriptive lengths (50–90 words).

### 4.4 Nucleus (Top-P) Sampling
- **Definition:** A stochastic decoding algorithm that dynamically restricts the candidate token pool to the smallest subset whose cumulative probability exceeds threshold $p$.
- **Formula:** $\sum_{w \in \mathcal{V}^{(p)}} P(w \mid y_{<t}) \ge p$
- **Descra Context:** Used when creative variations of product descriptions are desired (e.g., $p = 0.90$).

### 4.5 Repetition Penalty (`no_repeat_ngram_size`)
- **Definition:** A constraint that sets the logit probability of any candidate token to $-\infty$ if emitting it would duplicate an $n$-gram previously generated in the sequence.
- **Descra Context:** Set to `no_repeat_ngram_size: 3` in Descra, completely eliminating repetitive phrasing like "hydrating skin, hydrating skin".

### 4.6 Temperature
- **Definition:** A hyperparameter scaling factor applied to unnormalized logits before the softmax operation during stochastic sampling.
- **Formula:** $P(y_t = i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$
- **Descra Context:** Values $T < 1.0$ (e.g., $0.7$) produce conservative, factual text; values $T > 1.0$ produce diverse, creative variations.

---

## 5. Evaluation Metrics & Academic Rigor

### 5.1 BERTScore
- **Definition:** An automatic evaluation metric for text generation that computes semantic similarity between hypothesis and reference tokens using cosine similarity over contextualized embeddings (e.g., DeBERTa).
- **Descra Context:** Crucial for Descra because it recognizes valid synonym substitutions (e.g., "gentle formula" $\approx$ "mild ingredients") that BLEU and ROUGE penalize.

### 5.2 BLEU (Bilingual Evaluation Understudy)
- **Definition:** A precision-based metric calculating modified n-gram precision between a generated hypothesis and reference text, modulated by a Brevity Penalty (BP).
- **Descra Context:** Tracked across BLEU-1 through BLEU-4 to evaluate exact phrase preservation from reference descriptions.

### 5.3 Perplexity (PPL)
- **Definition:** The exponentiated cross-entropy loss, representing the effective branching factor of the model's predictive distribution: $\text{PPL} = \exp(\mathcal{L}_{\text{NLL}})$.
- **Descra Context:** Lower perplexity on the validation split indicates the model is less "surprised" by real-world human product descriptions.

### 5.4 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **Definition:** A recall-oriented metric measuring the proportion of reference n-grams or longest common subsequences found in the generated hypothesis.
- **Sub-metrics:**
  - **ROUGE-1:** Unigram recall (vocabulary coverage).
  - **ROUGE-2:** Bigram recall (attribute phrase preservation).
  - **ROUGE-L:** Longest Common Subsequence (sentence structure and syntactic flow).
- **Descra Context:** The primary quantitative benchmark for Phase 1 model selection. Target threshold: ROUGE-1 $\ge 0.38$, ROUGE-L $\ge 0.30$.

---

## 6. Descra-Specific Architecture & Engineering Concepts

### 6.1 Canonical Schema
- **Definition:** The standardized, strongly typed JSON structure into which raw Amazon metadata records are transformed during Step 3 preprocessing.
- **Fields:** `parent_asin`, `title`, `features`, `specifications`, `description`, `alignment_status`, `split`.

### 6.2 Factual Alignment
- **Definition:** The ratio of non-stopword content tokens in the reference description that are directly supported by the input title, bullet features, or specifications.
- **Descra Context:** Evaluated in Step 3; products with score $\ge 0.5$ are marked `ALIGNED`, $[0.2, 0.5)$ are `PARTIALLY_ALIGNED`, and $< 0.2$ are routed to `needs_review.jsonl`.

### 6.3 Hallucination
- **Definition:** The generation of statements, product benefits, or technical claims that cannot be grounded in or verified by the source input features.
- **Descra Context:** Mitigated by training only on factually aligned records and enforcing cross-attention conditioning during beam search decoding.

### 6.4 Prompt Formatter (`format_model_input`)
- **Definition:** The deterministic transformation function converting structured product metadata into an instruction-tuned input string for FLAN-T5.
- **Example:**
  `Generate product description: Title: Cetaphil Daily Hydrating Lotion | Key Features: 24 hour hydration; hyaluronic acid formula | Specifications: Skin Type: Dry, Sensitive`

