import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# 1. Config
DATA_PATH = "data/train.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/train.csv"

# 2. Feature Extraction
df = pd.read_csv(DATA_PATH).fillna("")
prompts = df['prompt'].str.lower().tolist()

vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(prompts)

features = []
labels = df['answer'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}).values

for idx, row in df.iterrows():
    p_vec = vectorizer.transform([row['prompt'].lower()])
    row_feats = []
    for choice in ['A', 'B', 'C', 'D', 'E']:
        c_vec = vectorizer.transform([str(row[choice]).lower()])
        sim = cosine_similarity(p_vec, c_vec)[0][0]
        row_feats.append(sim)
    features.append(row_feats)

X = np.array(features)
y = labels

# 3. Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=100,
        random_seed=42
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    val_acc = np.mean(model.predict(X_val).flatten() == y_val)
    print(f"Fold {fold} Val Accuracy: {val_acc:.4f}")
    break 