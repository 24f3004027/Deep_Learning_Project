# Error Analysis & Model Insights

This document outlines the failure modes, error classifications, and insights discovered during the evaluation of the deep learning and ensemble pipelines on the multiple-choice question dataset.

---

## 1. Classification of Model Errors
Through validation testing, we identified three major categories of questions where our sequence classification models (BiGRU and BERT) struggle:

### 1. Negation and Inverse Logic
*   **Description**: Questions containing words like *NOT*, *EXCEPT*, or *INCORRECT* (e.g., *"Which of the following is NOT a property of metals?"*).
*   **Error Pattern**: Deep learning models often focus on the strong semantic match between the positive option text (e.g. *"conducts electricity"*) and the prompt keywords, failing to register the negation operator. This results in the model predicting the exact opposite of the correct answer.

### 2. Numerical and Symbolic Reasoning
*   **Description**: Questions requiring mathematical calculations or ordering (e.g., *"Rank the following elements by increasing atomic radius"*).
*   **Error Pattern**: Since our vocabulary maps text words and does not perform active arithmetic computation, the model fails to evaluate numerical orderings correctly. It relies on text similarity which is insufficient for exact mathematical checks.

### 3. Semantic Ambiguity in Scientific Terms
*   **Description**: Questions where the correct answer requires deep domain knowledge that is not represented in the small training set vocabulary (e.g. specialized medical or quantum physics terms).
*   **Error Pattern**: The vocabulary maps these words to `<UNK>` (Unknown) tokens. The model is forced to predict based on surrounding function words (prepositions, conjunctions) which yields low confidence predictions.

---

## 2. The Lexical Distractor Trap
In scientific exams, test writers design "distractors" (incorrect choices) that contain a high density of keywords from the prompt to trick students. 

Our analysis shows that **tabular similarity models** (XGBoost/CatBoost) are highly susceptible to this trap:
*   Because tabular features measure raw overlap (TF-IDF cosine similarity, word matching counts), they assign a high score to distractors that copy-paste prompt keywords.
*   **Example**: 
    *   *Prompt*: "How does photosynthesis convert solar energy?"
    *   *Incorrect Distractor (High Overlap)*: "Photosynthesis is a process that converts solar energy into heat without creating sugars." (Lexical similarity is high, but the statement is factually false).
    *   *Correct Option (Lower Overlap)*: "It stores light energy in the chemical bonds of glucose molecules."
*   **Result**: The tabular model incorrectly ranks the distractor 1st. The semantic BiGRU model successfully ranks the correct option 1st by processing the context of "glucose" and "chemical bonds".

---

## 3. Mitigation Strategies & Insights

### 1. Blending NN and Tabular Models
By ensembling the models (60% Neural Network + 40% Boosters), we balance the NN's semantic logic with the Boosters' keyword matching. This corrects edge cases where the NN lacks vocabulary but the Booster identifies an exact option string match.

### 2. Test-Time Augmentation (TTA)
Applying TTA (averaging predictions of the original prompt and an instruction-augmented prompt) stabilizes predictions. It changes the attention maps of the transformer layers, forcing the model to focus on the options' details rather than just the question's keywords.