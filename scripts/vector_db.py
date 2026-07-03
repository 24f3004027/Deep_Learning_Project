import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

class SimpleVectorDB:
    def __init__(self, method="tfidf"):
        self.method = method
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        
    def fit(self, corpus):
        """
        Fits the retriever on a corpus of text strings.
        """
        # Deduplicate and clean documents
        self.documents = list(set([doc.strip() for doc in corpus if isinstance(doc, str) and len(doc.strip()) > 10]))
        
        if not self.documents:
            print("Warning: Fit corpus is empty!")
            return
            
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
            
    def query(self, text, k=3):
        """
        Queries the database and returns the top k relevant documents.
        """
        if not self.documents:
            return []
            
        if self.method == "tfidf" and self.vectorizer is not None:
            query_vector = self.vectorizer.transform([text])
            sims = cosine_similarity(query_vector, self.doc_vectors)[0]
            top_indices = np.argsort(sims)[::-1][:k]
            # Return documents and similarity scores
            return [self.documents[idx] for idx in top_indices if sims[idx] > 0.0]
            
        return []

def build_corpus_from_df(df):
    """
    Creates a corpus of documents from the prompt, choices, and answer columns of a DataFrame.
    """
    corpus = []
    for idx, row in df.iterrows():
        if isinstance(row['prompt'], str):
            corpus.append(row['prompt'])
        for choice in ['A', 'B', 'C', 'D', 'E']:
            if choice in row and isinstance(row[choice], str):
                corpus.append(row[choice])
    return corpus

def augment_prompt_with_context(prompt, context_list):
    """
    Combines retrieved context with the original prompt.
    """
    if not context_list:
        return prompt
    context_str = " ".join(context_list)
    return f"Context: {context_str}\nQuestion: {prompt}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()
    
    if args.test:
        # Create a mock corpus
        mock_corpus = [
            "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "Einstein's theory of relativity states that the laws of physics are the same for all non-accelerating observers.",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
            "The capital of France is Paris, which is known for its art, fashion, and culture.",
            "A black hole is a region of spacetime where gravity is so strong that nothing can escape."
        ]
        
        db = SimpleVectorDB(method="tfidf")
        db.fit(mock_corpus)
        
        test_queries = [
            "How do plants make energy?",
            "What did Einstein say about physics?",
            "At what temperature does water boil?"
        ]
        
        for q in test_queries:
            results = db.query(q, k=1)
            print(f"Query: '{q}'")
            print(f"Retrieved Context: '{results}'")
            print(f"Augmented Prompt:\n{augment_prompt_with_context(q, results)}")
            print("-" * 50)
            
        print("Vector DB test completed successfully!")
