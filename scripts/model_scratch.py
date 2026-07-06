import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import wandb

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "<SEP>"

class MCQVocabulary:
    def __init__(self, max_vocab_size=15000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SEP_TOKEN}
        
    def fit(self, texts):
        word_counts = {}
        for text in texts:
            if not isinstance(text, str):
                continue
            cleaned = re.sub(r"[^\w\s]", "", text.lower())
            for word in cleaned.split():
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:self.max_vocab_size]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    def encode(self, text, max_len=128):
        if not isinstance(text, str):
            return [0] * max_len
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        tokens = cleaned.split()
        encoded = [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in tokens]
        
        # Truncate
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        # Pad
        else:
            encoded = encoded + [self.word2idx[PAD_TOKEN]] * (max_len - len(encoded))
        return encoded

class ScratchMCQDataset(Dataset):
    def __init__(self, df, vocab, max_len=128):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
        
        # Format: for each choice, we concatenate prompt + SEP + choice
        input_ids = []
        for choice in ['A', 'B', 'C', 'D', 'E']:
            choice_text = str(row[choice])
            combined_text = f"{prompt} {SEP_TOKEN} {choice_text}"
            encoded = self.vocab.encode(combined_text, max_len=self.max_len)
            input_ids.append(encoded)
            
        input_ids = torch.tensor(input_ids, dtype=torch.long) # Shape: [5, max_len]
        
        if 'answer' in row:
            label = torch.tensor(self.label_map[row['answer']], dtype=torch.long)
            return input_ids, label
        return input_ids

class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, rnn_outputs):
        # rnn_outputs shape: [batch_size, seq_len, hidden_dim]
        weights = self.attention(rnn_outputs) # Shape: [batch_size, seq_len, 1]
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(rnn_outputs * weights, dim=1) # Shape: [batch_size, hidden_dim]
        return pooled

class BiGRUAttentionMCQModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.attention = SelfAttentionPooling(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, input_ids):
        # input_ids shape: [batch_size, 5, seq_len]
        batch_size, num_choices, seq_len = input_ids.shape
        
        # Flatten input to pass through recurrent layers
        # Flat shape: [batch_size * 5, seq_len]
        flat_input = input_ids.view(batch_size * num_choices, seq_len)
        
        embedded = self.embedding(flat_input) # [batch_size * 5, seq_len, embed_dim]
        rnn_out, _ = self.gru(embedded) # [batch_size * 5, seq_len, hidden_dim * 2]
        
        pooled = self.attention(rnn_out) # [batch_size * 5, hidden_dim * 2]
        logits = self.classifier(pooled) # [batch_size * 5, 1]
        
        # Reshape back to [batch_size, 5]
        logits = logits.view(batch_size, num_choices)
        return logits

def train_scratch_model(data_path, epochs=10, batch_size=32, lr=0.001, run_wandb=False):
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    df = pd.read_csv(data_path)
    # Basic data handling
    df = df.fillna("")
    
    # Vocabulary building
    corpus = df['prompt'].tolist()
    for col in ['A', 'B', 'C', 'D', 'E']:
        corpus.extend(df[col].tolist())
        
    vocab = MCQVocabulary()
    vocab.fit(corpus)
    vocab_size = len(vocab.word2idx)
    print(f"Vocabulary fitted. Total unique words: {vocab_size}")
    
    # Train / Validation Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['answer'])
    
    train_dataset = ScratchMCQDataset(train_df, vocab, max_len=128)
    val_dataset = ScratchMCQDataset(val_df, vocab, max_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BiGRUAttentionMCQModel(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if run_wandb:
        wandb.init(project="scratch-model-t22026", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "vocab_size": vocab_size,
            "architecture": "BiGRU + Self-Attention"
        })
        
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
            
        train_acc = accuracy_score(train_targets, train_preds)
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                val_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_loss /= len(val_loader)
        
        # Calculate mAP@3
        predictions_map3 = []
        target_map3 = []
        label_letters = ['A', 'B', 'C', 'D', 'E']
        for i, probs in enumerate(val_probs):
            sorted_indices = np.argsort(probs)[::-1][:3]
            preds = [label_letters[idx] for idx in sorted_indices]
            predictions_map3.append(preds)
            target_map3.append(label_letters[val_targets[i]])
            
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'vocab_size': vocab_size
            }, "checkpoints/scratch_model.pt")
            print("=> Saved new best model checkpoint!")
            
    if run_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        print("Testing Scratch Model architecture:")
        vocab = MCQVocabulary()
        vocab.fit(["sample query text", "option A", "option B", "option C", "option D", "option E"])
        model = BiGRUAttentionMCQModel(vocab_size=len(vocab.word2idx))
        # Batch size of 2, 5 choices, sequence length of 16
        dummy_input = torch.randint(0, len(vocab.word2idx), (2, 5, 16))
        out = model(dummy_input)
        print("Model output shape:", out.shape)
        assert out.shape == (2, 5)
        print("Scratch Model validation test passed!")
        
    elif args.train:
        train_scratch_model(args.data_path, epochs=args.epochs, run_wandb=args.wandb)
