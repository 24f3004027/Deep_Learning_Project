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

# Initialize theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Function to toggle theme state
def toggle_theme():
    if st.session_state.theme == "dark":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"

# Define colors for Light and Dark modes
if st.session_state.theme == "dark":
    bg_gradient = "linear-gradient(rgba(11, 15, 25, 0.88), rgba(11, 15, 25, 0.88))"
    text_color = "#f3f4f6"
    placeholder_color = "rgba(255, 255, 255, 0.45)"
    input_bg = "rgba(17, 24, 39, 0.9)"
    border_color = "rgba(255, 255, 255, 0.08)"
    card_bg = "rgba(255, 255, 255, 0.04)"
    card_border = "rgba(255, 255, 255, 0.07)"
    card_shadow = "rgba(0, 0, 0, 0.4)"
    link_color = "#3b82f6"
else:
    bg_gradient = "linear-gradient(rgba(243, 244, 246, 0.85), rgba(243, 244, 246, 0.85))"
    text_color = "#111827"
    placeholder_color = "rgba(17, 24, 39, 0.55)"
    input_bg = "rgba(255, 255, 255, 0.95)"
    border_color = "rgba(0, 0, 0, 0.15)"
    card_bg = "rgba(255, 255, 255, 0.8)"
    card_border = "rgba(0, 0, 0, 0.1)"
    card_shadow = "rgba(0, 0, 0, 0.08)"
    link_color = "#1d4ed8"

# CSS injection to style all labels, headers, text areas, placeholder texts, and background animation
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Apply background to main app containers and run pan animation */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-image: {bg_gradient}, url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1600') !important;
        background-size: 180% 180% !important; /* Scale up to enable panning range */
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: {text_color} !important;
        animation: pan-bg 45s ease-in-out infinite alternate !important; /* Forces background movement */
    }}
    
    @keyframes pan-bg {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Text Input & Area boxes */
    .stTextArea textarea, .stTextInput input {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 2px solid {border_color} !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
    }}
    
    /* PLACEHOLDER VISIBILITY FIX: Forces placeholder text to stay visible in both themes */
    textarea::placeholder, input::placeholder, .stTextArea textarea::placeholder, .stTextInput input::placeholder {{
        color: {placeholder_color} !important;
        opacity: 1 !important;
    }}
    
    /* STUBBORN LABELS FIX: Forces light and dark modes to show labels correctly */
    label, p, span, h1, h2, h3, h5, div[data-testid="stWidgetLabel"] p, .stTextArea label, .stTextInput label, div[data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
        font-weight: 600 !important;
    }}
    
    /* Buttons */
    button[kind="primary"] {{
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.3s ease !important;
    }}
    button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    }}
    
    /* Glassmorphic card design */
    .glass-card {{
        background: {card_bg} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid {card_border} !important;
        border-radius: 18px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px 0 {card_shadow} !important;
        margin-top: 15px !important;
        margin-bottom: 20px !important;
        animation: fadeIn 0.8s ease-in-out;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("🧠 Smart MCQ Solver Challenge")
st.write("##### *Made as a part of the Deep Learning & Generative AI Course*")

# Author Info Card
st.markdown(f"""
<div class="glass-card" style="padding: 15px 20px !important; margin-bottom: 20px !important;">
    <table style="width: 100%; border: none; margin: 0;">
        <tr style="background: none; border: none;">
            <td style="border: none; padding: 0; font-weight: 600; color: {text_color} !important;">👤 Made by: <b>Ramrup Satpati</b></td>
            <td style="border: none; padding: 0; text-align: right; font-weight: 600; color: {text_color} !important;">🆔 Roll Number: <b>24f3004027</b></td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# LinkedIn Link Card (replaces the old empty grey spacer)
st.markdown(f"""
<div class="glass-card" style="padding: 15px 20px !important; margin-bottom: 25px !important; text-align: center;">
    🔗 <b>Connect with me:</b> <a href="https://www.linkedin.com/in/ramrup-satpati-683970341/" target="_blank" style="color: {link_color} !important; font-weight: 700; text-decoration: none;">Go to my LinkedIn Profile</a>
</div>
""", unsafe_allow_html=True)

# Theme Toggle Button
theme_emoji = "☀️" if st.session_state.theme == "dark" else "🌙"
theme_label = "Switch to Light Mode" if st.session_state.theme == "dark" else "Switch to Dark Mode"
st.button(f"{theme_emoji} {theme_label}", on_click=toggle_theme)

# Input Forms (Clean layout, no empty HTML wrapper card)
prompt = st.text_area("Question / Prompt", placeholder="Type your multiple choice question here...", height=100)
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A", value="", placeholder="Enter choice A...")
    opt_b = st.text_input("Option B", value="", placeholder="Enter choice B...")
with col2:
    opt_c = st.text_input("Option C", value="", placeholder="Enter choice C...")
    opt_d = st.text_input("Option D", value="", placeholder="Enter choice D...")
opt_e = st.text_input("Option E", value="", placeholder="Enter choice E...")

submit = st.button("Analyze and Predict", type="primary")

# Process prediction
if submit:
    # 🌟 CRITICAL: If inputs are empty, show a warning notification and stop execution
    if not prompt.strip() or not all([opt_a.strip(), opt_b.strip(), opt_c.strip(), opt_d.strip(), opt_e.strip()]):
        st.error("⚠️ Please fill out the question prompt and all five option choices before analyzing!")
    else:
        choices = [opt_a, opt_b, opt_c, opt_d, opt_e]
        choice_labels = ['A', 'B', 'C', 'D', 'E']
        
        weights_path = "model_scratch_42.pt"
        vocab_path = "vocab.npy"
        
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
                
                # Render results in a single cohesive HTML card, preventing empty spacer bar issue
                best_idx = np.argmax(probs)
                results_card_html = f"""
                <div class="glass-card" style="border-left: 5px solid #2563eb !important; margin-top: 25px;">
                    <h3 style="color: {text_color} !important; margin-top: 0; margin-bottom: 15px;">📊 Neural Network Prediction Results</h3>
                    <div style="background-color: rgba(37, 99, 235, 0.12); border: 1px solid #2563eb; border-radius: 10px; padding: 15px; margin: 15px 0; color: {text_color} !important; font-weight: 600;">
                        ✅ Recommended Answer: Option {choice_labels[best_idx]} with {probs[best_idx]:.2%} confidence.
                    </div>
                    <p style="margin-bottom: 8px; color: {text_color} !important;">Option A: <b>{probs[0]:.2%}</b></p>
                    <p style="margin-bottom: 8px; color: {text_color} !important;">Option B: <b>{probs[1]:.2%}</b></p>
                    <p style="margin-bottom: 8px; color: {text_color} !important;">Option C: <b>{probs[2]:.2%}</b></p>
                    <p style="margin-bottom: 8px; color: {text_color} !important;">Option D: <b>{probs[3]:.2%}</b></p>
                    <p style="margin-bottom: 8px; color: {text_color} !important;">Option E: <b>{probs[4]:.2%}</b></p>
                </div>
                """
                st.markdown(results_card_html, unsafe_allow_html=True)
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
                
            # Render results in a single cohesive HTML card, preventing empty spacer bar issue
            best_idx = np.argmax(probs)
            results_card_html = f"""
            <div class="glass-card" style="border-left: 5px solid #2563eb !important; margin-top: 25px;">
                <h3 style="color: {text_color} !important; margin-top: 0; margin-bottom: 15px;">📊 Semantic Search Prediction Results (Fallback)</h3>
                <div style="background-color: rgba(37, 99, 235, 0.12); border: 1px solid #2563eb; border-radius: 10px; padding: 15px; margin: 15px 0; color: {text_color} !important; font-weight: 600;">
                    ✅ Recommended Answer: Option {choice_labels[best_idx]} with {probs[best_idx]:.2%} relative match score.
                </div>
                <p style="margin-bottom: 8px; color: {text_color} !important;">Option A: <b>{probs[0]:.2%}</b></p>
                <p style="margin-bottom: 8px; color: {text_color} !important;">Option B: <b>{probs[1]:.2%}</b></p>
                <p style="margin-bottom: 8px; color: {text_color} !important;">Option C: <b>{probs[2]:.2%}</b></p>
                <p style="margin-bottom: 8px; color: {text_color} !important;">Option D: <b>{probs[3]:.2%}</b></p>
                <p style="margin-bottom: 8px; color: {text_color} !important;">Option E: <b>{probs[4]:.2%}</b></p>
            </div>
            """
            st.markdown(results_card_html, unsafe_allow_html=True)

# Footer Section (Copyleft GPL Footer)
st.markdown(f"""
<hr style="border-color: {border_color}; margin-top: 50px;">
<div style="text-align: center; font-size: 0.85em; opacity: 0.7; padding: 10px 0; color: {text_color} !important;">
    🄯 | Copyleft Ramrup Satpati (2026) | All Rights Reversed | Released under the GNU General Public License
</div>
""", unsafe_allow_html=True)
