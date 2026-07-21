# Technical Report: Smart MCQ Solver Challenge
**Course Project: Deep Learning & Generative AI**  
**Roll Number:** 24f3004027  
**W&B Project Dashboard Link:** [Weights & Biases Dashboard](https://wandb.ai/24f3004027-indian-institute-of-technology-madras/24f3004027-t22026?nw=nwuser24f3004027)  

---

## 1. Problem Statement & Formulation
The objective of this challenge is to solve multiple-choice science questions by choosing the correct answer from five options (A, B, C, D, E). The dataset contains structured questions with a `prompt` (question body) and five text columns corresponding to options `A` through `E`. The target variable is `answer`, containing values in `{"A", "B", "C", "D", "E"}`.

We formulate this task as a **Multiple-Choice Classification Problem** where the input consists of a question-option pair. For each question, the model evaluates five separate sequences, producing five logits:
\[ \mathbf{z} = [z_A, z_B, z_C, z_D, z_E] \]
The option index with the highest logit value is predicted as the correct answer. The models are trained using Multi-Class Cross-Entropy loss over the ground truth choice distribution.

---

## 2. Methodology & Model Architectures
To satisfy the term project requirements, we designed, evaluated, and tracked three distinct architectures:

### Model 1: Bidirectional GRU with Self-Attention (Built from Scratch)
We built a custom sequence classification pipeline from scratch to avoid dependency on heavy pre-trained weights.
*   **Preprocessing**: Text is lowercased, punctuation is stripped to reduce vocabulary sparsity, and whitespaces are normalized.
*   **Embedding Layer**: An trainable PyTorch Embedding layer of dimension 128 maps the custom vocabulary.
*   **Recurrent Layer**: A bidirectional single-layer Gated Recurrent Unit (GRU) with a hidden size of 128 processes the sequence, capturing left-to-right and right-to-left contextual semantics.
*   **Self-Attention Pooling**: Rather than using simple max/average pooling, we implement a trainable self-attention head:
    \[ \alpha_t = \text{softmax}(\mathbf{w}^\top \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{b}_h)) \]
    This calculates a weighted sum of the hidden states, pooling them into a dense sequence representation.
*   **Classification Head**: A Linear-ReLU-Dropout-Linear classifier predicts logit scores for each option pair.

### Model 2: Fine-Tuned transformer (Pre-trained Model)
We utilize `bert-base-uncased` to leverage deep contextual embeddings trained on large corpora.
*   **Formatting**: Each choice is concatenated as `[CLS] Prompt [SEP] Option [SEP]` and processed by the BERT tokenizer.
*   **Forward Pass**: The 5 sequences are passed concurrently through `AutoModelForMultipleChoice`. BERT extracts context-aware representations, pooling them via the classification token representation.
*   **Classifier**: A linear projection maps the hidden dimension (768) to a single scalar logit for each choice.

### Model 3: Tabular-DL Hybrid Ensemble (Model of Choice)
To reduce variance and leverage the distinct strengths of statistical and semantic models, we developed a three-way hybrid ensemble:
1.  **5-Seed PyTorch BiGRU Average (60% weight)**: A blend of 5 BiGRU models trained with different seeds (42, 123, 2026, 888, 999) on 100% of the training dataset.
2.  **Deep XGBoost Classifier (20% weight)**: A tree-based model trained on 25 hand-crafted features extracted for the five choices.
3.  **Deep CatBoost Classifier (20% weight)**: Trained on the same 25 tabular features.

#### Feature Engineering for Tabular Models (25 Features total):
For each choice in a question, we extract:
*   **TF-IDF Cosine Similarity**: Lexical alignment between prompt and option vectors.
*   **Word Overlap Ratio**: Normalized count of intersecting words.
*   **Jaccard Similarity**: Word intersection over union.
*   **Choice Character Length**: Absolute length of option string.
*   **Length Ratio**: Length of option relative to the prompt length.

---

## 3. Training Details & Hyperparameters
To ensure reproducibility, the models were trained under the following parameters:

| Parameter | Model 1 (Scratch) | Model 2 (BERT) | Model 3 (Ensemble - Boosters) |
| :--- | :--- | :--- | :--- |
| **Batch Size** | 32 | 8 | N/A (Tabular matrix) |
| **Optimizer** | AdamW (`weight_decay=1e-2`) | AdamW (`weight_decay=1e-2`) | XGBoost / CatBoost |
| **Learning Rate** | 1e-3 | 2e-5 | 0.01 (XGB) / 0.01 (Cat) |
| **Epochs / Trees** | 5 epochs | 3 epochs | 15000 trees / 3000 iterations |
| **Sequence Length** | 128 tokens | 128 tokens | N/A |
| **Early Stopping** | None | None | 150 rounds validation wait |

> [!NOTE]
> During Model 3 training, the tabular models were trained on a 90/10 train-validation split. XGBoost triggered early stopping at tree **2855** (validation loss `0.109`), and CatBoost converged at iteration **2999** (validation loss `0.167`).

---

## 4. Evaluation & Leaderboard Performance
Below is a summary of the model evaluation runs compared against the Kaggle public leaderboard score:

| Run / Model | Config Details | Val Accuracy (Local) | Leaderboard Score |
| :--- | :--- | :--- | :--- |
| **V4 Baseline** | Compact BiGRU, Seed 42, split train | 0.7700 | **0.75727** |
| **V15 Custom LSTM** | Custom BiLSTM with shape error | Crash | N/A |
| **V16 Custom LSTM** | Fixed BiLSTM, 2-layer, hidden 256 | 0.6900 | 0.74231 |
| **V17 Regularized** | BiGRU, learning rate scheduler | 0.7350 | 0.74688 |
| **V20 Tabular Blend** | 5-seed BiGRU + XGB/Cat Ensemble | 0.8120 | 0.74812 |

---

## 5. Error Analysis & Key Insights

### Insight 1: The "Distractor Trap" (Lexical Overlap Bias)
A major finding of our experiment was that ensembling tabular boosters (XGBoost/CatBoost) using TF-IDF and overlap features actually **lowered** the Kaggle test score (from `0.757` down to `0.748`), despite showing a high local validation score (`0.812`).
*   **The Cause**: In high-quality science MCQs, distractors (incorrect choices) are deliberately written to contain high lexical overlap and similar scientific keywords to deceive student reasoning.
*   **The Impact**: The tabular classifiers relied purely on keyword overlap, falling straight into the distractor traps. The semantic BiGRU model, processing word sequences and contextual relations, bypassed these traps and outperformed the ensembled model.

### Insight 2: Target Leakage Prevention in RAG
When implementing Retrieval-Augmented Generation (RAG) on the training set, we discovered severe target leakage:
*   *Without masking*: The prompt queries the index and retrieves its own ground-truth option at rank #1. The model learns to match the context exactly, achieving 100% training accuracy but failing entirely on the test set.
*   *With masking*: We explicitly mask out the current question index during training, forcing the retriever to fetch secondary related facts. This ensures the model learns robust contextual reasoning.

---

## 6. Conclusion & Future Directions
Our exploration shows that for complex MCQ tasks, semantic sequence matching models generalize better than tabular overlap boosters due to distractor keyword traps. To improve performance further, future research will explore:
1.  **Deeper Context Embeddings**: Using domain-specific models like SciBERT to extract features.
2.  **RAG Pipelines**: Integrating multi-hop retrieval from general scientific corpuses (e.g. Wikipedia).
