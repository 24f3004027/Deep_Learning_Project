# Project Proposal: Smart MCQ Solver Challenge

## 1. Objectives & Goals
The primary objective of this project is to build an end-to-end multiple-choice question solver capable of predicting the correct answer from five options (A, B, C, D, E). The model must rank the top three options for each question to optimize the Mean Average Precision at 3 (MAP@3) metric.

## 2. Dataset Properties
The dataset is provided as part of the IITM course project competition:
*   **Training Set**: 2,000 question samples. Each row contains a question prompt, five choices (A, B, C, D, E), and the correct answer column.
*   **Test Set**: 500 question samples (without the answer column, used for leaderboard evaluation).
*   **Features**: Textual prompt, choice options, and index mappings.

## 3. Evaluation Metric (MAP@3)
The evaluation metric is Mean Average Precision at 3 (MAP@3), defined as:
\[ \text{MAP@3} = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{\min(3, n)} P(k) \times \text{rel}(k) \]
Where:
*   \(U\) is the total number of questions.
*   \(P(k)\) is the precision at cut-off \(k\).
*   \(\text{rel}(k)\) is an indicator function equal to 1 if item at rank \(k\) is correct, 0 otherwise.