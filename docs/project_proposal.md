# Project Proposal: Smart MCQ Solver Challenge

## 1. Introduction & Background
Solving multiple-choice questions (MCQs) in the scientific domain represents a fundamental benchmark for Natural Language Processing (NLP) and Artificial Intelligence. Unlike standard text classification, multiple-choice question answering requires models to perform semantic alignment, logical deduction, and contextual matching between a question body (the prompt) and multiple candidates.

The "Smart MCQ Solver Challenge" presents a dataset of scientific and general knowledge questions. The goal is to develop a robust machine learning system that ranks the five available options (A to E) and outputs the top three candidates in order of descending probability.

---

## 2. Dataset Profile & Structure
The dataset consists of structured text columns. The data is partitioned into:
*   **Training Set (`train.csv`)**: 2,000 rows. Includes a unique identifier (`id`), the question body (`prompt`), five candidate choices (`A`, `B`, `C`, `D`, `E`), and the ground-truth label (`answer`).
*   **Test Set (`test.csv`)**: 500 rows. Formatted identically but excludes the `answer` column. It is utilized to generate the final competition predictions.

### Data Columns:
1.  `id`: A unique string or numeric identifier.
2.  `prompt`: The text representing the question body.
3.  `A`, `B`, `C`, `D`, `E`: The text options representing the choices.
4.  `answer`: The correct option label (A, B, C, D, or E).

---

## 3. Problem Formulation
We formulate this task as a **Sequence Pair Classification and Ranking Task**.
For a given prompt \( P \) and options \( \{O_A, O_B, O_C, O_D, O_E\} \):
1.  We construct five input sequences by concatenating the prompt with each option using a separator token:
    \[ S_i = P \oplus \text{[SEP]} \oplus O_i \quad \text{for } i \in \{A, B, C, D, E\} \]
2.  The model processes each sequence \( S_i \) to generate a raw logit score \( z_i \).
3.  Softmax is applied across all five options to construct a probability distribution:
    \[ P(O_i | P) = \frac{e^{z_i}}{\sum_{j \in \{A,B,C,D,E\}} e^{z_j}} \]
4.  The options are sorted by probability in descending order, and the top three indices are output.

---

## 4. Evaluation Metric (MAP@3)
The model performance is evaluated using **Mean Average Precision at 3 (MAP@3)**. This metric penalizes models for placing the correct answer lower in their list of predictions.

The mathematical formulation for a single user query (question) is:
\[ \text{AP@3} = \sum_{k=1}^{3} P(k) \times \text{rel}(k) \]
Where:
*   \( P(k) \) is the precision at cut-off \( k \): the proportion of recommended items up to rank \( k \) that are correct.
*   \( \text{rel}(k) \) is an indicator function equal to 1 if the item at rank \( k \) is the correct answer, and 0 otherwise.

### MAP@3 Scoring Scenarios:
Since there is only one correct answer per question, the Average Precision (AP@3) for a single row resolves to:
*   **1.0000** if the correct option is ranked 1st (e.g., predicted: **C** A B, ground truth: **C**).
*   **0.5000** if the correct option is ranked 2nd (e.g., predicted: A **C** B, ground truth: **C**).
*   **0.3333** if the correct option is ranked 3rd (e.g., predicted: A B **C**, ground truth: **C**).
*   **0.0000** if the correct option is not in the Top 3 predictions.

The Mean Average Precision (MAP@3) is the mean of the AP@3 scores across all \( U \) evaluated questions:
\[ \text{MAP@3} = \frac{1}{U} \sum_{u=1}^{U} \text{AP@3}_u \]

---

## 5. Technical Challenges & Strategy

### The Lexical Distractor Trap
A major challenge in scientific multiple-choice question answering is the presence of "distractors"—incorrect options designed specifically to deceive models. These distractors frequently contain correct scientific keywords or high exact-word overlap with the prompt.
*   **Lexical Models**: Simple models relying on TF-IDF cosine similarity or word count matching often rank these distractors first because of high keyword overlap.
*   **Contextual Models**: To bypass this trap, we implement deep semantic models (BiGRU with Self-Attention, BERT, DeBERTa) that process sequence grammar, context, and negations, rather than relying on raw keyword frequency.

### Retrieval-Augmented Generation (RAG)
To provide the classification model with relevant scientific domain knowledge, we implement an offline RAG pipeline. For each prompt, we query a vector database containing scientific facts to retrieve the top 2 context documents, prepending this context to the prompt before classification.