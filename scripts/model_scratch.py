import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

# Import modular text helper
from clean_tokenize import clean_text, MCQVocabulary, PAD_TOKEN, UNK_TOKEN, SEP_TOKEN

# Attempt Weights & Biases import for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
TEST_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv"

# Scan paths dynamically
for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        if file == 'train.csv':
            TRAIN_PATH = os.path.join(root, file)
        elif file == 'test.csv':
            TEST_PATH = os.path.join(root, file)

# Fallback path scan locally
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/Users/ramrupsatpati/Documents/DL_GENAI/data/train.csv"
    TEST_PATH = "/Users/ramrupsatpati/Documents/DL_GENAI/data/test.csv"

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
EMBED_DIM = 128
HIDDEN_DIM = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------
# Dataset Class
# ----------------------------------------------------
class ScratchMCQDataset(Dataset):
    def __init__(self, df, vocab, max_len=MAX_LEN, is_train=True):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
       
    def __len__(self):
        return len(self.df)
       
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
       
        input_ids = []
        for choice in ['A', 'B', 'C', 'D', 'E']:
            choice_text = str(row[choice])
            combined_text = f"{prompt} {SEP_TOKEN} {choice_text}"
            encoded = self.vocab.encode(combined_text, max_len=self.max_len)
            input_ids.append(encoded)
           
        input_ids = torch.tensor(input_ids, dtype=torch.long)
       
        if self.is_train and 'answer' in row:
            label = torch.tensor(self.label_map[row['answer']], dtype=torch.long)
            return input_ids, label
        return input_ids

# ----------------------------------------------------
# Architecture (BiGRU - 1 Layer)
# ----------------------------------------------------
class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
       
    def forward(self, rnn_outputs):
        weights = self.attention(rnn_outputs)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(rnn_outputs * weights, dim=1)
        return pooled

class BiGRUAttentionMCQModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = SelfAttentionPooling(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
       
    def forward(self, input_ids):
        batch_size, num_choices, seq_len = input_ids.shape
        flat_input = input_ids.view(batch_size * num_choices, seq_len)
       
        embedded = self.embedding(flat_input)
        rnn_out, _ = self.gru(embedded)
       
        pooled = self.attention(rnn_out)
        logits = self.classifier(pooled)
       
        return logits.view(batch_size, num_choices)

# ----------------------------------------------------
# Main Training Logic
# ----------------------------------------------------
def main():
    print(f"Loading data from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH).fillna("")
    test_df = pd.read_csv(TEST_PATH).fillna("")
    
    # Fit Vocab
    corpus = train_df['prompt'].tolist()
    for col in ['A', 'B', 'C', 'D', 'E']:
        corpus.extend(train_df[col].tolist())
    vocab = MCQVocabulary()
    vocab.fit(corpus)
    vocab_size = len(vocab.word2idx)
    print(f"Vocabulary fitted. Size: {vocab_size}")
    
    # Train / Val Split (using 0.15 size)
    train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['answer'])
    
    train_dataset = ScratchMCQDataset(train_data, vocab, is_train=True)
    val_dataset = ScratchMCQDataset(val_data, vocab, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    seed = 42
    set_seed(seed)
    print(f"Training Model 1 (Scratch BiGRU) on Seed {seed}...")
    
    # WandB tracking
    if WANDB_AVAILABLE:
        wandb.init(project="24f3004027-t22026", name=f"Scratch-BiGRU-Seed-{seed}", config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM
        })
        
    model = BiGRUAttentionMCQModel(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        train_acc = correct / total
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
    if WANDB_AVAILABLE:
        wandb.finish()
        
    print("Training finished.")

if __name__ == "__main__":
    main()
