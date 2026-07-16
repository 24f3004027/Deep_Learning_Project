# Deep Learning & Generative AI Project (D&G Project)

This repository contains the code and notebooks for the DL & GenAI course project (Diploma Level of BS in Data Science and Applications).

## Project Details
*   **Project Title**: Context-Augmented Multiple-Choice QA using Deep Learning and RAG
*   **Name**: [Your Name]
*   **Roll No / Student ID**: [Your Roll No] (W&B Project: `[YourRollNo]-t22026`)
*   **Kaggle Notebook**: `DL-[YourRollNo]-notebook-t22026`

---

## Directory Structure
```
DL_GENAI/
├── .venv/                      # Python virtual environment
├── requirements.txt            # Package dependencies
├── README.md                   # Project details and execution guide
├── data/                       # Datasets
│   ├── train.csv               # MCQ training set
│   ├── test.csv                # MCQ test set
│   └── sample_submission.csv   # Kaggle sample submission file
├── scripts/                    # Modular Python code for reproducibility
│   ├── __init__.py
│   ├── clean_tokenize.py       # Text cleaning, tokenization & baseline similarity (Milestone 1)
│   ├── vector_db.py            # Simple TF-IDF Vector Database for RAG (Milestone 3)
│   ├── model_scratch.py        # Model 1: Custom PyTorch BiGRU + Attention (from scratch)
│   ├── model_pretrained.py     # Model 2: Fine-tuned DistilBERT (pretrained)
│   ├── model_choice.py         # Model 3: DeBERTa-v3-small + LoRA PEFT + RAG (choice model)
│   └── ensemble.py             # Predictions ensembling and CSV formatting (Milestone 5)
└── notebooks/                  # Interactive notebooks corresponding to milestones
    ├── milestone_1_nlp.ipynb   # Milestone 1: Classical similarity baselines
    ├── milestone_2_trans.ipynb # Milestone 2: Transformer-based embeddings
    ├── milestone_3_rag.ipynb   # Milestone 3: Retrieval Augmentation Demonstration
    ├── milestone_4_ft.ipynb    # Milestone 4: Fine-tuning setups and logs
    ├── milestone_5_ens.ipynb   # Milestone 5: Predictions ensembling
    └── DL-t22026-notebook.ipynb # Combined Kaggle Submission Notebook
```

---

## Environment Setup

1.  **Create and activate the virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## Training and Running Models

All model scripts are runnable via the command line. You can train them individually or run tests to verify their architectures.

### 1. Model 1: Custom PyTorch BiGRU + Attention (Built from Scratch)
*   **Verify architecture**:
    ```bash
    python3 scripts/model_scratch.py --test
    ```
*   **Train model**:
    ```bash
    python3 scripts/model_scratch.py --train --epochs 5 --wandb
    ```

### 2. Model 2: Fine-Tuned DistilBERT (Pre-trained Model)
*   **Verify architecture**:
    ```bash
    python3 scripts/model_pretrained.py --test
    ```
*   **Train model**:
    ```bash
    python3 scripts/model_pretrained.py --train --epochs 3 --wandb
    ```

### 3. Model 3: DeBERTa-v3 + LoRA PEFT + RAG (Choice Model)
*   **Verify architecture**:
    ```bash
    python3 scripts/model_choice.py --test
    ```
*   **Train model**:
    ```bash
    python3 scripts/model_choice.py --train --epochs 3 --wandb
    ```

---

## Ensembling & Final Submission

Once the models are trained and checkpoints are saved in the `checkpoints/` directory, you can run the ensemble inference script to generate the final submission file:
```bash
python3 scripts/ensemble.py --test_csv data/test.csv --output_csv submission.csv
```
This will create a `submission.csv` containing your ensembled predictions, ready to be submitted to Kaggle.

---

## Experiment Tracking
All models are integrated with Weights & Biases (W&B) for experiment tracking. Make sure to log in to W&B before training:
```bash
wandb login
```
Each run will track losses, accuracies, and F1 scores, allowing easy side-by-side comparison on the W&B dashboard.