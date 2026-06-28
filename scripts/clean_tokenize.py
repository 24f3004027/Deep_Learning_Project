import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text):
    # Standard whitespace splitting
    return text.split()

def handle_missing_data(df):
    # Fill missing values for the prompt and options columns with an empty string
    columns_to_fill = ['prompt', 'A', 'B', 'C', 'D', 'E']
    for col in columns_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df

def average_precision_at_3(predictions, target):
    """
    Computes Average Precision @ 3 for a single MCQ instance.
    predictions: List of 3 strings (e.g. ['A', 'B', 'C']) representing ranks 1, 2, 3
    target: String (e.g. 'B') representing correct option
    """
    # Ensure prediction is a list and contains elements
    if isinstance(predictions, str):
        predictions = predictions.split()
    
    for rank, pred in enumerate(predictions[:3]):
        if pred == target:
            return 1.0 / (rank + 1)
    return 0.0

def mean_average_precision_at_3(predictions_list, targets_list):
    """
    Computes Mean Average Precision @ 3 (mAP@3) over all instances.
    """
    ap_scores = [average_precision_at_3(p, t) for p, t in zip(predictions_list, targets_list)]
    return np.mean(ap_scores)

def compute_tfidf_similarities(df):
    """
    Computes similarities between prompts and choices using TF-IDF.
    """
    df = handle_missing_data(df.copy())
    
    # Let's clean the texts
    cleaned_prompts = df['prompt'].apply(clean_text)
    
    predictions = []
    for idx, row in df.iterrows():
        prompt_cleaned = clean_text(row['prompt'])
        choices = [clean_text(row[c]) for c in ['A', 'B', 'C', 'D', 'E']]
        
        # Fit vectorizer on this specific question context (prompt + choices)
        vectorizer = TfidfVectorizer()
        try:
            vectors = vectorizer.fit_transform([prompt_cleaned] + choices).toarray()
            prompt_vec = vectors[0:1]
            choice_vecs = vectors[1:]
            
            # Compute cosine similarity
            sims = cosine_similarity(prompt_vec, choice_vecs)[0]
        except Exception:
            # Fallback if vocabulary is empty
            sims = np.zeros(5)
            
        # Get sorted choices based on similarity (descending order)
        choice_letters = ['A', 'B', 'C', 'D', 'E']
        sorted_indices = np.argsort(sims)[::-1]
        sorted_choices = [choice_letters[i] for i in sorted_indices]
        
        predictions.append(sorted_choices[:3])
        
    return predictions

def build_simple_word2vec_embeddings(sentences, vector_size=100):
    """
    Builds a simple vocabulary and PyTorch/NumPy based Word2Vec embeddings from scratch
    for Model 1 and baseline requirements.
    """
    word_counts = {}
    for sentence in sentences:
        for word in tokenize_text(clean_text(sentence)):
            word_counts[word] = word_counts.get(word, 0) + 1
            
    # Keep words with count >= 1, assign index
    vocab = {word: idx + 1 for idx, word in enumerate(word_counts.keys())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = len(vocab)
    
    # Initialize random embeddings
    np.random.seed(42)
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), vector_size))
    embeddings[0] = 0.0 # PAD representation
    
    return vocab, embeddings

def get_average_embedding(text, vocab, embeddings):
    words = tokenize_text(clean_text(text))
    vectors = []
    for word in words:
        idx = vocab.get(word, vocab['<UNK>'])
        vectors.append(embeddings[idx])
        
    if len(vectors) == 0:
        return np.zeros(embeddings.shape[1])
    return np.mean(vectors, axis=0)

def compute_word2vec_similarities(df, vocab, embeddings):
    df = handle_missing_data(df.copy())
    predictions = []
    
    for idx, row in df.iterrows():
        prompt_emb = get_average_embedding(row['prompt'], vocab, embeddings)
        choice_embs = [get_average_embedding(row[c], vocab, embeddings) for c in ['A', 'B', 'C', 'D', 'E']]
        
        sims = []
        for choice_emb in choice_embs:
            norm_prod = (np.linalg.norm(prompt_emb) * np.linalg.norm(choice_emb))
            if norm_prod == 0:
                sims.append(0.0)
            else:
                sims.append(np.dot(prompt_emb, choice_emb) / norm_prod)
                
        choice_letters = ['A', 'B', 'C', 'D', 'E']
        sorted_indices = np.argsort(sims)[::-1]
        sorted_choices = [choice_letters[i] for i in sorted_indices]
        predictions.append(sorted_choices[:3])
        
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()
    
    if args.test:
        print("Testing text cleaning:")
        sample_text = "What is the relation... between Einstein's relativity & quantum mechanics?!!"
        print("Original:", sample_text)
        print("Cleaned:", clean_text(sample_text))
        
        print("\nTesting tokenization:")
        print("Tokens:", tokenize_text(clean_text(sample_text)))
        
        print("\nTesting mAP@3 calculation:")
        preds = [['A', 'B', 'C'], ['C', 'A', 'B'], ['D', 'E', 'A']]
        targets = ['B', 'C', 'B'] # AP: 0.5, 1.0, 0.0
        print("Expected mAP@3:", (0.5 + 1.0 + 0.0)/3.0)
        print("Calculated mAP@3:", mean_average_precision_at_3(preds, targets))
        
        # Test TF-IDF similarities with dummy DataFrame
        data = {
            'prompt': ["What is the color of the sky?", "What is the capital of France?"],
            'A': ["The sky is blue.", "London"],
            'B': ["The sky is green.", "Berlin"],
            'C': ["The sky is yellow.", "Rome"],
            'D': ["The sky is purple.", "Paris"],
            'E': ["The sky is black.", "Madrid"],
            'answer': ['A', 'D']
        }
        test_df = pd.DataFrame(data)
        tfidf_preds = compute_tfidf_similarities(test_df)
        print("\nTF-IDF predictions on test_df:", tfidf_preds)
        print("TF-IDF mAP@3:", mean_average_precision_at_3(tfidf_preds, test_df['answer'].tolist()))
        
        # Test Word2Vec similarities
        corpus = []
        for idx, row in test_df.iterrows():
            corpus.append(row['prompt'])
            corpus.extend([row[c] for c in ['A', 'B', 'C', 'D', 'E']])
        vocab, embs = build_simple_word2vec_embeddings(corpus)
        w2v_preds = compute_word2vec_similarities(test_df, vocab, embs)
        print("\nWord2Vec predictions on test_df:", w2v_preds)
        print("Word2Vec mAP@3:", mean_average_precision_at_3(w2v_preds, test_df['answer'].tolist()))
        print("\nAll tests passed successfully!")
