import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import sys
import wandb

# Add path for local scripts imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clean_tokenize import mean_average_precision_at_3
from vector_db import SimpleVectorDB, build_corpus_from_df, augment_prompt_with_context

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
INDEX_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

class RAGMCQDataset(Dataset):
    def __init__(self, df, tokenizer, vector_db, max_len=192, k_contexts=2):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vector_db = vector_db
        self.max_len = max_len
        self.k_contexts = k_contexts
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
        
        # Retrieve context from Vector DB (excluding exact prompt match to avoid direct leakage)
        # In a real RAG setup we would query an external database. Here we query the train corpus.
        retrieved_contexts = self.vector_db.query(prompt, k=self.k_contexts + 1)
        # Filter out the exact prompt or very similar text if retrieved
        filtered_contexts = [ctx for ctx in retrieved_contexts if ctx.strip().lower() != prompt.strip().lower()][:self.k_contexts]
        
        # Prepend context to prompt
        augmented_prompt = augment_prompt_with_context(prompt, filtered_contexts)
        
        # Format input prompt + choices
        first_sentences = [augmented_prompt] * 5
        second_sentences = [str(row[option]) for option in ['A', 'B', 'C', 'D', 'E']]
        
        # Tokenize prompt + option pairs
        encoded = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        item = {key: val for key, val in encoded.items()}
        for key in item:
            item[key] = item[key].squeeze(0)
            
        if 'answer' in row:
            item['labels'] = torch.tensor(LABEL_MAP[row['answer']], dtype=torch.long)
            
        return item

def train_choice_model(data_path, model_name="microsoft/deberta-v3-small", epochs=3, batch_size=2, lr=5e-5, run_wandb=False):
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    df = pd.read_csv(data_path)
    df = df.fillna("")
    
    # Build RAG corpus and fit Vector DB
    print("Building RAG corpus and indexing...")
    corpus = build_corpus_from_df(df)
    vector_db = SimpleVectorDB(method="tfidf")
    vector_db.fit(corpus)
    print("RAG database fitted successfully!")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['answer'])
    
    # Device
    # Use CPU on Darwin/Mac specifically for DeBERTa due to known numerical instability/NaN bugs on MPS
    import platform
    if "deberta" in model_name.lower() and platform.system() == "Darwin":
        device = torch.device("cpu")
        print("Running DeBERTa on CPU locally: downsampling training dataset to 150 train and 30 validation rows for fast execution.")
        train_df = train_df.sample(n=150, random_state=42)
        val_df = val_df.sample(n=30, random_state=42)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = RAGMCQDataset(train_df, tokenizer, vector_db, max_len=192, k_contexts=2)
    val_dataset = RAGMCQDataset(val_df, tokenizer, vector_db, max_len=192, k_contexts=2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    base_model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    # LoRA PEFT Configuration
    # We target query and value projections which are standard for Transformer attention blocks
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    # Wrap model with LoRA adapters
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model = model.float()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    if run_wandb:
        wandb.init(project="choice-model-t22026", config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lora_r": 8,
            "lora_alpha": 16,
            "k_contexts": 2,
            "architecture": "DeBERTa-v3-small + LoRA + RAG"
        })
        
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(train_targets, train_preds)
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_loss /= len(val_loader)
        
        # Calculate mAP@3
        predictions_map3 = []
        target_map3 = []
        for i, probs in enumerate(val_probs):
            sorted_indices = np.argsort(probs)[::-1][:3]
            preds = [INDEX_MAP[idx] for idx in sorted_indices]
            predictions_map3.append(preds)
            target_map3.append(INDEX_MAP[val_targets[i]])
            
        val_map3 = mean_average_precision_at_3(predictions_map3, target_map3)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val mAP@3: {val_map3:.4f}")
        
        if run_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_map3": val_map3
            })
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            model.save_pretrained("checkpoints/choice_model")
            tokenizer.save_pretrained("checkpoints/choice_model")
            print("=> Saved new best choice model!")
            
    if run_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        print("Testing Choice Model (DeBERTa + LoRA + RAG):")
        # Initialize small corpus for testing
        df_test = pd.DataFrame([{
            'prompt': "Is quantum computing real?",
            'A': "Yes, using qubits",
            'B': "No",
            'C': "Maybe",
            'D': "It is a dream",
            'E': "It is classic",
            'answer': 'A'
        }])
        vector_db = SimpleVectorDB(method="tfidf")
        vector_db.fit(["quantum computing is real and uses qubits."])
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
        dataset = RAGMCQDataset(df_test, tokenizer, vector_db, max_len=64)
        loader = DataLoader(dataset, batch_size=1)
        batch = next(iter(loader))
        
        print("Batch input_ids shape:", batch['input_ids'].shape)
        assert batch['input_ids'].shape == (1, 5, 64)
        
        # Load small multiple choice model to test Peft model wrapping
        base_model = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-small")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=4,
            lora_alpha=8,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
        
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        print("Logits shape:", outputs.logits.shape)
        assert outputs.logits.shape == (1, 5)
        print("Choice model architecture validation passed!")
        
    elif args.train:
        train_choice_model(args.data_path, model_name=args.model_name, epochs=args.epochs, run_wandb=args.wandb)
