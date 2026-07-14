import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import PeftModel
import argparse
import sys

# Add path for local scripts imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_scratch import BiGRUAttentionMCQModel, ScratchMCQDataset, MCQVocabulary
from model_pretrained import HFMCQDataset
from model_choice import RAGMCQDataset
from vector_db import SimpleVectorDB, build_corpus_from_df

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
INDEX_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

def run_ensemble_inference(test_csv_path, output_csv_path="submission.csv", w1=0.2, w2=0.4, w3=0.4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.fillna("")
    
    # ------------------ Model 1: Scratch BiGRU ------------------
    model_scratch_path = "checkpoints/scratch_model.pt"
    probs_scratch = None
    if os.path.exists(model_scratch_path):
        print("Loading Model 1 (Scratch BiGRU)...")
        checkpoint = torch.load(model_scratch_path, map_location=device, weights_only=False)
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        
        model_1 = BiGRUAttentionMCQModel(vocab_size=vocab_size)
        model_1.load_state_dict(checkpoint['model_state_dict'])
        model_1 = model_1.to(device)
        model_1.eval()
        
        dataset_1 = ScratchMCQDataset(test_df, vocab, max_len=128)
        loader_1 = DataLoader(dataset_1, batch_size=16, shuffle=False)
        
        probs_scratch_list = []
        with torch.no_grad():
            for batch_x in loader_1:
                batch_x = batch_x.to(device)
                logits = model_1(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_scratch_list.append(probs)
        probs_scratch = np.concatenate(probs_scratch_list, axis=0)
        print("Model 1 inference complete!")
    else:
        print("Warning: Model 1 checkpoint not found. Skipping model 1.")
        w1 = 0.0

    # ------------------ Model 2: Pretrained DistilBERT ------------------
    model_pretrained_dir = "checkpoints/pretrained_model"
    probs_pretrained = None
    if os.path.exists(model_pretrained_dir):
        print("Loading Model 2 (Pretrained DistilBERT)...")
        tokenizer_2 = AutoTokenizer.from_pretrained(model_pretrained_dir)
        model_2 = AutoModelForMultipleChoice.from_pretrained(model_pretrained_dir).to(device)
        model_2.eval()
        
        dataset_2 = HFMCQDataset(test_df, tokenizer_2, max_len=128)
        loader_2 = DataLoader(dataset_2, batch_size=8, shuffle=False)
        
        probs_pretrained_list = []
        with torch.no_grad():
            for batch in loader_2:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                probs_pretrained_list.append(probs)
        probs_pretrained = np.concatenate(probs_pretrained_list, axis=0)
        print("Model 2 inference complete!")
    else:
        print("Warning: Model 2 checkpoint not found. Skipping model 2.")
        w2 = 0.0

    # ------------------ Model 3: Choice Model (DeBERTa + LoRA + RAG) ------------------
    model_choice_dir = "checkpoints/choice_model"
    probs_choice = None
    if os.path.exists(model_choice_dir):
        print("Loading Model 3 (LoRA DeBERTa-v3 + RAG)...")
        tokenizer_3 = AutoTokenizer.from_pretrained(model_choice_dir)
        
        # Re-build RAG Vector DB using training dataset
        train_path = "data/train.csv"
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path).fillna("")
            corpus = build_corpus_from_df(train_df)
            vector_db = SimpleVectorDB(method="tfidf")
            vector_db.fit(corpus)
        else:
            print("Warning: RAG training corpus not found. Running contextless query.")
            vector_db = SimpleVectorDB(method="tfidf") # empty
            
        dataset_3 = RAGMCQDataset(test_df, tokenizer_3, vector_db, max_len=192, k_contexts=2)
        loader_3 = DataLoader(dataset_3, batch_size=4, shuffle=False)
        
        # Load DeBERTa base model and wrap it with PEFT
        import platform
        choice_device = torch.device("cpu") if platform.system() == "Darwin" else device
        base_model = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-small")
        model_3 = PeftModel.from_pretrained(base_model, model_choice_dir).to(choice_device)
        model_3.eval()
        
        probs_choice_list = []
        with torch.no_grad():
            for batch in loader_3:
                input_ids = batch['input_ids'].to(choice_device)
                attention_mask = batch['attention_mask'].to(choice_device)
                
                outputs = model_3(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                probs_choice_list.append(probs)
        probs_choice = np.concatenate(probs_choice_list, axis=0)
        print("Model 3 inference complete!")
    else:
        print("Warning: Model 3 checkpoint not found. Skipping model 3.")
        w3 = 0.0

    # Normalize weights
    total_w = w1 + w2 + w3
    if total_w == 0:
        raise ValueError("Error: No valid model checkpoints were found. Please train models first!")
        
    w1, w2, w3 = w1 / total_w, w2 / total_w, w3 / total_w
    print(f"Combining models with weights -> Model1: {w1:.2f}, Model2: {w2:.2f}, Model3: {w3:.2f}")
    
    # Calculate ensembled probabilities
    num_samples = len(test_df)
    ensembled_probs = np.zeros((num_samples, 5))
    if probs_scratch is not None:
        ensembled_probs += w1 * probs_scratch
    if probs_pretrained is not None:
        ensembled_probs += w2 * probs_pretrained
    if probs_choice is not None:
        ensembled_probs += w3 * probs_choice
        
    # Generate final Predictions
    predictions = []
    for probs in ensembled_probs:
        sorted_indices = np.argsort(probs)[::-1][:3]
        prediction_str = " ".join([INDEX_MAP[idx] for idx in sorted_indices])
        predictions.append(prediction_str)
        
    # Save to submission CSV
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Prediction': predictions
    })
    
    submission_df.to_csv(output_csv_path, index=False)
    print(f"Submission saved successfully to {output_csv_path}!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--w1", type=float, default=0.2, help="Weight for scratch model")
    parser.add_argument("--w2", type=float, default=0.4, help="Weight for pretrained model")
    parser.add_argument("--w3", type=float, default=0.4, help="Weight for choice model")
    args = parser.parse_args()
    
    run_ensemble_inference(args.test_csv, args.output_csv, args.w1, args.w2, args.w3)
