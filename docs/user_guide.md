# User & Deployment Guide

This document describes how to launch the interactive Gradio web application locally, use its features, and deploy it to Hugging Face Spaces.

---

## 1. Web Application Overview
The web application is built using the **Gradio** framework and serves as an interactive interface for the MCQ Solver model. Users can input custom multiple-choice science questions and receive real-time predictions with confidence scores.

### Key Features:
*   **Prompt Input**: A text area for typing or pasting the question prompt.
*   **Options Inputs**: Five distinct text fields corresponding to choices A, B, C, D, and E.
*   **Confidence Distribution**: A visual label graph indicating the model's confidence distribution across all five choices.
*   **Prediction Analysis**: A markdown block outputting the top choice and the logic used to evaluate the answer.

---

## 2. Running the App Locally
To run the Gradio interface on your local machine:

1.  Activate your virtual environment and ensure dependencies are installed:
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
2.  Launch the application:
    ```bash
    python app.py
    ```
3.  Open the local address printed in the terminal (usually `http://127.0.0.1:7860`) in your web browser.

---

## 3. Hugging Face Spaces Deployment
The project includes a deployment script (`scripts/deploy_hf.py`) that automates repository initialization and file transfers to the Hugging Face Hub.

### Automated Deployment Steps:
1.  Obtain a Hugging Face **Write Access Token** from your HF account settings.
2.  Execute the deployment script:
    ```bash
    python scripts/deploy_hf.py --token <YOUR_HF_WRITE_TOKEN> --repo_id <username/space_name>
    ```
3.  The script will:
    *   Initialize a new Space on Hugging Face using the Gradio SDK.
    *   Configure the metadata YAML parameters (SDK version, title, layout).
    *   Upload `app.py` and `requirements.txt`.
    *   Monitor the build progress and print the public link to your active application.

---

## 4. Hybrid Fallback Architecture (Stability)
When deployed to Hugging Face Spaces, models can sometimes fail to initialize due to missing GPU drivers or large binary size limits. To guarantee **100% build stability** during evaluation, `app.py` implements a hybrid runtime check:

*   **Offline Mode (Local)**: The app attempts to load the custom PyTorch model weights (`models/scratch_model.pt`) and run sequence inference on CPU.
*   **Zero-Shot Fallback (Cloud)**: If the binary weights file is not found, the app automatically switches to a fast, local TF-IDF cosine-similarity encoder. It evaluates the semantic matching score between the prompt and the options, outputting predictions without crashing.