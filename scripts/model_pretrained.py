import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import wandb

# Mapping of labels to indices
LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
INDEX_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

class HFMCQDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
        
        # We need to construct prompt + option pair for each of the 5 options
        first_sentences = [prompt] * 5
        second_sentences = [str(row[option]) for option in ['A', 'B', 'C', 'D', 'E']]
        
        # Tokenize the pairs
        encoded = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # encoded features are tensors of shape [5, max_len]
        item = {key: val for key, val in encoded.items()}
        # Remove batch dimension added by return_tensors
        for key in item:
            item[key] = item[key].squeeze(0)
            
        if 'answer' in row:
            item['labels'] = torch.tensor(LABEL_MAP[row['answer']], dtype=torch.long)
            
        return item

def train_pretrained_model(data_path, model_name="distilbert-base-uncased", epochs=3, batch_size=4, lr=2e-5, run_wandb=False):
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    df = pd.read_csv(data_path)
    df = df.fillna("")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['answer'])
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = HFMCQDataset(train_df, tokenizer, max_len=128)
    val_dataset = HFMCQDataset(val_df, tokenizer, max_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    if run_wandb:
        wandb.init(project="pretrained-model-t22026", config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "architecture": "DistilBERT for Multiple Choice"
        })
        
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch in train_loader:
            # Move all tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
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
            
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from clean_tokenize import mean_average_precision_at_3
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
            model.save_pretrained("checkpoints/pretrained_model")
            tokenizer.save_pretrained("checkpoints/pretrained_model")
            print("=> Saved new best pretrained model!")
            
    if run_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        print("Testing pre-trained model setup:")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForMultipleChoice.from_pretrained("distilbert-base-uncased")
        
        # Test input formatting
        data = {
            'prompt': "What is gravity?",
            'A': "A force",
            'B': "An apple",
            'C': "A theory",
            'D': "A planet",
            'E': "A sound",
            'answer': 'A'
        }
        df_test = pd.DataFrame([data])
        dataset = HFMCQDataset(df_test, tokenizer, max_len=32)
        loader = DataLoader(dataset, batch_size=1)
        batch = next(iter(loader))
        
        print("Batch input_ids shape:", batch['input_ids'].shape)
        assert batch['input_ids'].shape == (1, 5, 32)
        
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        print("Logits shape:", outputs.logits.shape)
        assert outputs.logits.shape == (1, 5)
        print("Pretrained model architecture validation passed!")
        
    elif args.train:
        train_pretrained_model(args.data_path, model_name=args.model_name, epochs=args.epochs, run_wandb=args.wandb)