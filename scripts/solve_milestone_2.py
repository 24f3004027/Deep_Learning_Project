import os
import re
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForMultipleChoice, pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper to compute MAP@3
def average_precision_at_3(predictions, target):
    for rank, pred in enumerate(predictions[:3]):
        if pred == target:
            return 1.0 / (rank + 1)
    return 0.0

def mean_average_precision_at_3(predictions_list, targets_list):
    ap_scores = [average_precision_at_3(p, t) for p, t in zip(predictions_list, targets_list)]
    return np.mean(ap_scores)

# --- QUESTION 1 ---
print("--- Computing Question 1: HF datasets length of combined_text ---")
# Load dataset using HF Datasets library (do not use pandas)
dataset = load_dataset("csv", data_files="data/train.csv", split="train")

def concatenate_prompt_A(example):
    example["combined_text"] = f"{example['prompt']} {example['A']}"
    return example

dataset = dataset.map(concatenate_prompt_A)
row_51_combined = dataset[51]["combined_text"]
q1_length = len(row_51_combined)
print(f"Answer Q1 (Length at index 51): {q1_length}")

# --- QUESTION 2 ---
print("\n--- Computing Question 2: bert-base-uncased vocab size ---")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
print(f"Answer Q2 (Vocab size): {vocab_size}")

# --- QUESTION 3 ---
print("\n--- Computing Question 3: [SEP] token ID ---")
sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
print(f"Answer Q3 ([SEP] token ID): {sep_id}")

# --- QUESTION 4 ---
print("\n--- Computing Question 4: Shape of tokenized prompt ---")
prompts = [str(x) if x is not None else "" for x in dataset["prompt"]]
tokenized_prompts = tokenizer(
    prompts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
q4_shape = list(tokenized_prompts["input_ids"].shape)
print(f"Answer Q4 (Shape of input_ids): {q4_shape}")

# --- QUESTION 5 ---
print("\n--- Computing Question 5: Dimensionality of each attention head ---")
# Hidden size = 768, attention heads = 12
head_dim = 768 // 12
print(f"Answer Q5 (Attention head dim): {head_dim}")

# --- QUESTION 6 ---
print("\n--- Computing Question 6: Shape of last_hidden_state for row ID 0 ---")
# Row ID 0 is index 0 (since ID starts at 1, row ID 0 refers to the first row of the dataset)
row_0_prompt = dataset[0]["prompt"]
inputs_row_0 = tokenizer(row_0_prompt, return_tensors="pt")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()
with torch.no_grad():
    outputs_row_0 = model(**inputs_row_0)
q6_shape = list(outputs_row_0.last_hidden_state.shape)
print(f"Answer Q6 (Shape of last_hidden_state): {q6_shape}")

# --- QUESTION 7 ---
print("\n--- Computing Question 7: Sum of first 5 float values in [CLS] embedding ---")
cls_vector = outputs_row_0.last_hidden_state[0, 0].numpy()
first_5_sum = np.sum(cls_vector[:5])
q7_sum = round(float(first_5_sum), 4)
print(f"Answer Q7 (Sum of first 5 float values): {q7_sum}")

# --- QUESTION 8 ---
print("\n--- Computing Question 8: Attention weight CLS pays to fusion ---")
model_att = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
model_att.eval()

sentence = "Light-ion fusion is a technique."
inputs_att = tokenizer(sentence, return_tensors="pt")
input_ids_list = inputs_att["input_ids"][0].tolist()

# Find token index for "fusion"
tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
fusion_idx = None
for idx, token in enumerate(tokens):
    if token.lower() == "fusion":
        fusion_idx = idx
        break
print(f"Tokens: {tokens}")
print(f"Token index for 'fusion': {fusion_idx}")

with torch.no_grad():
    outputs_att = model_att(**inputs_att)

# Extract attention matrix for last layer (index -1) and first head (head index 0)
# Shape of attention tensor in BERT: [num_layers, batch_size, num_heads, seq_len, seq_len]
last_layer_att = outputs_att.attentions[-1] # Shape: [1, 12, seq_len, seq_len]
cls_to_fusion_weight = last_layer_att[0, 0, 0, fusion_idx].item()
q8_weight = round(cls_to_fusion_weight, 4)
print(f"Answer Q8 (Attention weight): {q8_weight}")

# --- QUESTION 9 ---
print("\n--- Computing Question 9: Sentence-transformers similarity for row ID 0 ---")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
row_0_opt_b = dataset[0]["B"]

prompt_emb = st_model.encode(row_0_prompt, convert_to_tensor=True)
opt_b_emb = st_model.encode(row_0_opt_b, convert_to_tensor=True)
sim_st = util.cos_sim(prompt_emb, opt_b_emb).item()
q9_sim = round(sim_st, 4)
print(f"Answer Q9 (Cosine similarity): {q9_sim}")

# --- QUESTION 10 ---
print("\n--- Computing Question 10: TF-IDF vs SentenceTransformer MAP@3 & overlapping count ---")
# Build Pipeline 1: TF-IDF cosine similarity from Milestone 1
train_df = pd.read_csv("data/train.csv").fillna("")

# Preprocessing
def clean_text_simple(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Running TF-IDF pipeline...")
tfidf_preds = []
for idx, row in train_df.iterrows():
    p_clean = clean_text_simple(row['prompt'])
    choices = [clean_text_simple(row[c]) for c in ['A', 'B', 'C', 'D', 'E']]
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([p_clean] + choices).toarray()
        sims = cosine_similarity(vectors[0:1], vectors[1:])[0]
    except Exception:
        sims = np.zeros(5)
    sorted_indices = np.argsort(sims)[::-1]
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    tfidf_preds.append([choice_letters[i] for i in sorted_indices[:3]])

# Build Pipeline 2: sentence-transformers/all-MiniLM-L6-v2
print("Running MiniLM pipeline...")
minilm_preds = []
prompts = train_df['prompt'].tolist()
# Encode prompts in batch for speed
prompt_embeddings = st_model.encode(prompts, show_progress_bar=True)

for idx, row in train_df.iterrows():
    p_emb = prompt_embeddings[idx]
    choices = [row[c] for c in ['A', 'B', 'C', 'D', 'E']]
    choice_embs = st_model.encode(choices)
    
    sims = []
    for c_emb in choice_embs:
        sims.append(util.cos_sim(p_emb, c_emb).item())
        
    sorted_indices = np.argsort(sims)[::-1]
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    minilm_preds.append([choice_letters[i] for i in sorted_indices[:3]])

targets = train_df['answer'].tolist()
minilm_map3 = mean_average_precision_at_3(minilm_preds, targets)
print(f"Answer Q10a (MiniLM MAP@3): {minilm_map3:.4f}")

# Count of correct answer NOT in TF-IDF Top-3 BUT IN MiniLM Top-3
count_overlap = 0
for idx, target in enumerate(targets):
    in_tfidf = target in tfidf_preds[idx]
    in_minilm = target in minilm_preds[idx]
    if (not in_tfidf) and in_minilm:
        count_overlap += 1
print(f"Answer Q10b (Overlap Count): {count_overlap}")

# --- QUESTION 11 ---
print("\n--- Computing Question 11: Zero-shot classification (Softmax) ---")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
row_1_prompt = dataset[1]["prompt"]
row_1_labels = [dataset[1]["A"], dataset[1]["B"], dataset[1]["C"]]

res_softmax = classifier(row_1_prompt, candidate_labels=row_1_labels, multi_label=False)
top_score_softmax = res_softmax["scores"][0]
q11_score = round(top_score_softmax, 4)
print(f"Softmax Result scores: {res_softmax['scores']}")
print(f"Answer Q11 (Top score Softmax): {q11_score}")

# --- QUESTION 12 ---
print("\n--- Computing Question 12: Zero-shot classification (Independent Sigmoids) ---")
res_sigmoid = classifier(row_1_prompt, candidate_labels=row_1_labels, multi_label=True)
sigmoid_scores = res_sigmoid["scores"]
print(f"Sigmoid Result scores: {sigmoid_scores}")

# Sum of Softmax vs Sum of Sigmoid scores
sum_softmax = sum(res_softmax["scores"])
sum_sigmoid = sum(sigmoid_scores)
abs_diff = abs(sum_softmax - sum_sigmoid)
q12_diff = round(abs_diff, 4)
print(f"Sum of Softmax: {sum_softmax}, Sum of Sigmoid: {sum_sigmoid}")
print(f"Answer Q12 (Absolute difference): {q12_diff}")

# --- QUESTION 13 ---
print("\n--- Computing Question 13: Text2Text generation with flan-t5-small ---")
generator = pipeline("text2text-generation", model="google/flan-t5-small")
row_0_prompt = dataset[0]["prompt"]
row_0_a = dataset[0]["A"]
row_0_b = dataset[0]["B"]

input_str = f"Question: {row_0_prompt}. Is the correct answer A: {row_0_a} or B: {row_0_b}? Answer with just the letter A or B."
res_t5 = generator(input_str, max_new_tokens=5)
q13_text = res_t5[0]["generated_text"].strip()
print(f"Answer Q13 (Generated Text): '{q13_text}'")
