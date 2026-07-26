import streamlit as st
import numpy as np
import re
import os
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define PAD, UNK, SEP tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "<SEP>"

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
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.set_page_config(page_title="Smart MCQ Solver", layout="centered")
st.title("🧠 Smart MCQ Solver Challenge")
st.subheader("Deep Learning & Generative AI Project (Roll: 24f3004027)")
st.write("Input a question prompt and its five choices to predict the correct answer option.")

prompt = st.text_area("Question / Prompt", placeholder="Type your multiple choice question here...", height=100)
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A", value="Option A text...")
    opt_b = st.text_input("Option B", value="Option B text...")
with col2:
    opt_c = st.text_input("Option C", value="Option C text...")
    opt_d = st.text_input("Option D", value="Option D text...")
opt_e = st.text_input("Option E", value="Option E text...")

if st.button("Analyze and Predict", type="primary"):
    choices = [opt_a, opt_b, opt_c, opt_d, opt_e]
    choice_labels = ['A', 'B', 'C', 'D', 'E']
    
    weights_path = "model_scratch_42.pt"
    vocab_path = "vocab.npy"
    
    # Check if we can run PyTorch NN model
    if os.path.exists(weights_path) and os.path.exists(vocab_path):
        try:
            word2idx = np.load(vocab_path, allow_pickle=True).item()
            vocab_size = len(word2idx)
            model = BiGRUAttentionMCQModel(vocab_size, embed_dim=128, hidden_dim=128)
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model.eval()
            
            def encode_text(text, max_len=128):
                cleaned = clean_text(text)
                tokens = cleaned.split()
                encoded = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in tokens]
                if len(encoded) > max_len:
                    encoded = encoded[:max_len]
                else:
                    encoded = encoded + [word2idx.get("<PAD>", 0)] * (max_len - len(encoded))
                return encoded

            input_ids = []
            for choice in choices:
                combined_text = f"{prompt} {SEP_TOKEN} {choice}"
                input_ids.append(encode_text(combined_text))
            
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_ids_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()
            
            st.markdown("### 📊 Neural Network Prediction Results")
            best_idx = np.argmax(probs)
            st.success(f"**Recommended Answer**: Option **{choice_labels[best_idx]}** with **{probs[best_idx]:.2%}** confidence.")
            for i in range(5):
                st.write(f"Option **{choice_labels[i]}**: {probs[i]:.2%}")
        except Exception as e:
            st.warning(f"Error loading PyTorch model: {e}. Falling back to TF-IDF matching...")
            st.stop()
    else:
        # Fallback to TF-IDF semantic match
        cleaned_prompt = clean_text(prompt)
        cleaned_choices = [clean_text(opt) for opt in choices]
        corpus = [cleaned_prompt] + cleaned_choices
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            p_vec = tfidf_matrix[0]
            c_vecs = tfidf_matrix[1:]
            
            sims = cosine_similarity(p_vec, c_vecs)[0]
            exp_sims = np.exp(sims * 5)
            probs = exp_sims / np.sum(exp_sims)
        except Exception:
            probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
        st.markdown("### 📊 Semantic Search Prediction Results (Fallback)")
        best_idx = np.argmax(probs)
        st.success(f"**Recommended Answer**: Option **{choice_labels[best_idx]}** with **{probs[best_idx]:.2%}** relative match score.")
        for i in range(5):
            st.write(f"Option **{choice_labels[i]}**: {probs[i]:.2%}")
