import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# 1. Config
DATA_PATH = "data/train.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/train.csv"

# 2. Hand-crafted similarity features
df = pd.read_csv(DATA_PATH).fillna("")
features = []
labels = df['answer'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}).values

for idx, row in df.iterrows():
    p_words = set(str(row['prompt']).lower().split())
    row_feats = []
    for choice in ['A', 'B', 'C', 'D', 'E']:
        c_words = set(str(row[choice]).lower().split())
        overlap = len(p_words.intersection(c_words))
        jaccard = len(p_words.intersection(c_words)) / (len(p_words.union(c_words)) + 1e-8)
        row_feats.extend([overlap, jaccard])
    features.append(row_feats)

X = np.array(features)
y = labels

# 3. Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    val_acc = np.mean(model.predict(X_val) == y_val)
    print(f"Fold {fold} Val Accuracy: {val_acc:.4f}")
    break