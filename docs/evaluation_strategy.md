# Evaluation Strategy & Experiment Tracking

This document outlines the validation scheme, performance metrics, and experiment tracking configurations used to train and compare the models for the Smart MCQ Solver Challenge.

---

## 1. Validation Split & Stratification
Due to the small size of the training dataset (2,000 samples), a robust validation split is required to prevent overfitting and measure generalization accurately.

*   **Split Ratio**: We implement an 85/15 train-validation split (1,700 rows for training, 300 rows for local validation).
*   **Stratification**: Multiple-choice labels (A to E) are not uniformly distributed in every subset. To prevent class imbalance shifts between our splits, we apply **Stratified Splitting** on the target `answer` column:
    ```python
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.15, 
        random_state=42, 
        stratify=train_df['answer']
    )
    ```
    This guarantees that the proportion of correct answers (A, B, C, D, E) remains identical in both the training and validation sets.

---

## 2. Evaluation Metrics

### 1. Classification Accuracy
Accuracy measures the ratio of correct predictions to the total samples:
\[ \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \]
We track local validation accuracy across all epochs to determine the optimal stopping point for Model 1 and Model 2.

### 2. Mean Average Precision at 3 (MAP@3)
Since the competition allows three ranked predictions (e.g. `C A B`), we evaluate predictions using MAP@3. The formula for the average precision of a single question is:
\[ \text{AP@3} = \sum_{k=1}^{3} P(k) \times \text{rel}(k) \]
Where:
*   AP@3 is 1.0 if the correct answer is the 1st guess.
*   AP@3 is 0.5 if the correct answer is the 2nd guess.
*   AP@3 is 0.33 if the correct answer is the 3rd guess.
*   AP@3 is 0.0 otherwise.

Local validation MAP@3 is calculated over the validation split to verify that ensembling and RAG augmentations improve the model's ranking ability.

---

## 3. Experiment Tracking (Weights & Biases)
To meet the project criteria of tracking and comparing at least three runs, we integrate **Weights & Biases (W&B)** into all training scripts.

### Logged Parameters:
For every model run, we log:
*   **Configuration Metadata**: Model architecture names, learning rates, weight decays, dropout rates, embedding dimensions, and sequence lengths.
*   **Metrics**: Train loss, validation loss, validation accuracy, and epoch steps.

### Dashboard Organization:
Runs are grouped under the project directory `24f3004027-t22026`. We construct comparative charts (Val Accuracy vs. Epochs) to visually contrast the models:
1.  `Scratch-BiGRU-Seed-42`: Evaluating custom embedding and attention pooling performance.
2.  `Pretrained-BERT`: Evaluating transformer transfer learning convergence.
3.  `Ensemble-NN-Seeds`: Evaluating variance reduction across different seeds.