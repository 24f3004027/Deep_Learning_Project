import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created notebook: {filename}")

# Helper to create cells
def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

os.makedirs("notebooks", exist_ok=True)

# ----------------- MILESTONE 1 NOTEBOOK -----------------
m1_cells = [
    markdown_cell("# Milestone 1: NLP Foundation & Semantic Similarity\n"
                  "This notebook covers the foundations of NLP for our Multiple-Choice QA task. We will clean text, tokenize, handle missing data, generate TF-IDF and custom Word2Vec representations, compute similarities, and evaluate our models using the Mean Average Precision @ 3 (mAP@3) metric."),
    code_cell("import os\nimport sys\nimport pandas as pd\nimport numpy as np\n\n# Add scripts path\nsys.path.append('../scripts')\nfrom clean_tokenize import clean_text, tokenize_text, handle_missing_data, mean_average_precision_at_3, compute_tfidf_similarities, build_simple_word2vec_embeddings, compute_word2vec_similarities"),
    markdown_cell("## 1. Load and Inspect Dataset"),
    code_cell("train_df = pd.read_csv('../data/train.csv')\nprint(f'Train set shape: {train_df.shape}')\ntrain_df.head()"),
    markdown_cell("## 2. Handling Missing Data and Cleaning"),
    code_cell("# Apply missing data handler\ntrain_df = handle_missing_data(train_df)\n\n# Sample text cleaning\nsample_prompt = train_df.iloc[0]['prompt']\ncleaned_prompt = clean_text(sample_prompt)\ntokens = tokenize_text(cleaned_prompt)\n\nprint('Original Prompt:', sample_prompt)\nprint('Cleaned Prompt:', cleaned_prompt)\nprint('Tokens:', tokens)"),
    markdown_cell("## 3. Cosine Similarity via TF-IDF Baseline\n"
                  "We will compute cosine similarity between each prompt and its corresponding options A, B, C, D, E using TF-IDF."),
    code_cell("print('Computing TF-IDF predictions...')\ntfidf_preds = compute_tfidf_similarities(train_df)\nprint('Sample TF-IDF prediction:', tfidf_preds[0])"),
    markdown_cell("## 4. Evaluate TF-IDF via mAP@3\n"
                  "Calculate the mAP@3 on the training set using the TF-IDF similarities."),
    code_cell("targets = train_df['answer'].tolist()\ntfidf_map3 = mean_average_precision_at_3(tfidf_preds, targets)\nprint(f'TF-IDF Baseline mAP@3 Score: {tfidf_map3:.4f}')"),
    markdown_cell("## 5. Cosine Similarity via Word2Vec Embeddings\n"
                  "Now we build a simple Word2Vec embedding mapping from scratch, compute average word vectors for prompts and options, and compute cosine similarities."),
    code_cell("print('Building custom Word2Vec embeddings from scratch...')\ncorpus = train_df['prompt'].tolist()\nfor col in ['A', 'B', 'C', 'D', 'E']:\n    corpus.extend(train_df[col].tolist())\n    \nvocab, embs = build_simple_word2vec_embeddings(corpus, vector_size=100)\n\nprint('Computing Word2Vec predictions...')\nw2v_preds = compute_word2vec_similarities(train_df, vocab, embs)\n\nw2v_map3 = mean_average_precision_at_3(w2v_preds, targets)\nprint(f'Word2Vec Baseline mAP@3 Score: {w2v_map3:.4f}')")
]
create_notebook("notebooks/milestone_1_nlp.ipynb", m1_cells)

# ----------------- MILESTONE 2 NOTEBOOK -----------------
m2_cells = [
    markdown_cell("# Milestone 2: Enter the Transformers\n"
                  "In this milestone, we introduce pre-trained transformer embeddings. We will extract contextual embeddings using standard transformer libraries and compute cosine similarities to evaluate how context-aware embeddings compare to classical baselines."),
    code_cell("import os\nimport sys\nimport pandas as pd\nimport numpy as np\nimport torch\nfrom transformers import AutoTokenizer, AutoModel\nfrom sklearn.metrics.pairwise import cosine_similarity\n\nsys.path.append('../scripts')\nfrom clean_tokenize import mean_average_precision_at_3"),
    markdown_cell("## 1. Extract context-aware embeddings using pre-trained DistilBERT"),
    code_cell("device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')\ntokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')\nmodel = AutoModel.from_pretrained('distilbert-base-uncased').to(device)\nmodel.eval()"),
    markdown_cell("## 2. Text Embeddings Extraction Helper"),
    code_cell("def get_transformer_embedding(text, max_len=128):\n    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)\n    with torch.no_grad():\n        outputs = model(**inputs)\n    # Mean pooling over token embeddings\n    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()\n    return embeddings[0]\n\nsample_emb = get_transformer_embedding('Sample text for transformer embedding.')\nprint('Embedding shape:', sample_emb.shape)"),
    markdown_cell("## 3. Predict Options via Cosine Similarity\n"
                  "Compute the similarity score between prompt embedding and option embeddings for the first 50 training instances."),
    code_cell("train_df = pd.read_csv('../data/train.csv').head(50)\npredictions = []\n\nfor idx, row in train_df.iterrows():\n    p_emb = get_transformer_embedding(row['prompt']).reshape(1, -1)\n    choice_embs = [get_transformer_embedding(row[c]).reshape(1, -1) for c in ['A', 'B', 'C', 'D', 'E']]\n    \n    sims = []\n    for c_emb in choice_embs:\n        sims.append(cosine_similarity(p_emb, c_emb)[0][0])\n        \n    choice_letters = ['A', 'B', 'C', 'D', 'E']\n    sorted_indices = np.argsort(sims)[::-1]\n    sorted_choices = [choice_letters[i] for i in sorted_indices]\n    predictions.append(sorted_choices[:3])\n\ntargets = train_df['answer'].tolist()\nmap3 = mean_average_precision_at_3(predictions, targets)\nprint(f'Transformer-Embedding Cosine Similarity mAP@3: {map3:.4f}')")
]
create_notebook("notebooks/milestone_2_trans.ipynb", m2_cells)

# ----------------- MILESTONE 3 NOTEBOOK -----------------
m3_cells = [
    markdown_cell("# Milestone 3: Context Augmentation with RAG Pipelines\n"
                  "This milestone implements a simple Retrieval-Augmented Generation (RAG) pipeline. General LLMs have context size limits and lack domain-specific details. We will build a simple vector database, index our training sentences, query relevant context, and prepend it to our prompts."),
    code_cell("import os\nimport sys\nimport pandas as pd\n\nsys.path.append('../scripts')\nfrom vector_db import SimpleVectorDB, build_corpus_from_df, augment_prompt_with_context"),
    markdown_cell("## 1. Load Training Data & Construct Vector DB"),
    code_cell("train_df = pd.read_csv('../data/train.csv')\n\n# Create a corpus from the dataset text\ncorpus = build_corpus_from_df(train_df)\nprint(f'Total documents in corpus for indexing: {len(corpus)}')\n\n# Initialize and fit TF-IDF based vector database\nvector_db = SimpleVectorDB(method='tfidf')\nvector_db.fit(corpus)"),
    markdown_cell("## 2. Demonstrate RAG Context Retrieval"),
    code_cell("sample_prompt = 'What is the relationship between the Wigner function and the density matrix?'\nretrieved = vector_db.query(sample_prompt, k=2)\nprint('Original Query:', sample_prompt)\nprint('\\nRetrieved Contexts:')\nfor i, doc in enumerate(retrieved):\n    print(f'{i+1}. {doc}')\n\nprint('\\nAugmented Prompt:')\nprint(augment_prompt_with_context(sample_prompt, retrieved))")
]
create_notebook("notebooks/milestone_3_rag.ipynb", m3_cells)

# ----------------- MILESTONE 4 NOTEBOOK -----------------
m4_cells = [
    markdown_cell("# Milestone 4: MCQ Formulation & Fine-Tuning\n"
                  "This notebook details the training setup, LoRA parameters, and W&B logging for the three Deep Learning models. Training commands are listed below for reproducibility."),
    code_cell("import os\nimport sys\nimport pandas as pd\n\n# Verify scripts are importable\nsys.path.append('../scripts')\nimport model_scratch\nimport model_pretrained\nimport model_choice"),
    markdown_cell("## Model MCQ Input Formulating\n"
                  "In multiple-choice classification, we formulate our input as 5 prompt-choice sequence pairs. We tokenise them and pass through the model to predict which choice is correct.\n"
                  "\n"
                  "### To train Model 1: Custom PyTorch BiGRU + Attention (Built from Scratch)\n"
                  "```bash\n"
                  "python3 ../scripts/model_scratch.py --train --epochs 5 --wandb\n"
                  "```\n"
                  "\n"
                  "### To train Model 2: Fine-Tuned DistilBERT (Pre-trained)\n"
                  "```bash\n"
                  "python3 ../scripts/model_pretrained.py --train --epochs 3 --wandb\n"
                  "```\n"
                  "\n"
                  "### To train Model 3: DeBERTa-v3 + LoRA PEFT + RAG (Choice Model)\n"
                  "```bash\n"
                  "python3 ../scripts/model_choice.py --train --epochs 3 --wandb\n"
                  "```"),
    markdown_cell("## 1. Verify Model Architectures via Local Tests"),
    code_cell("!python3 ../scripts/model_scratch.py --test\n!python3 ../scripts/model_pretrained.py --test\n!python3 ../scripts/model_choice.py --test")
]
create_notebook("notebooks/milestone_4_ft.ipynb", m4_cells)

# ----------------- MILESTONE 5 NOTEBOOK -----------------
m5_cells = [
    markdown_cell("# Milestone 5: Ensembling & Predictions\n"
                  "This notebook demonstrates loading the checkpoints from the three trained models, getting predictions on the test set, ensembling predictions using a weighted average of normalized logits, sorting the logits to output the top 3 predictions, and saving the submission."),
    code_cell("import os\nimport sys\nimport pandas as pd\n\nsys.path.append('../scripts')\nimport ensemble"),
    markdown_cell("## 1. Run Ensemble Inference on Test Dataset"),
    code_cell("# Verify if test dataset exists\ntest_path = '../data/test.csv'\nif os.path.exists(test_path):\n    # We combine predictions using weights: w1=0.2 (Scratch), w2=0.4 (DistilBERT), w3=0.4 (DeBERTa+LoRA)\n    print('Running ensembled inference...')\n    ensemble.run_ensemble_inference(test_path, output_csv_path='../submission.csv', w1=0.2, w2=0.4, w3=0.4)\nelse:\n    print('Test CSV not found!')"),
    markdown_cell("## 2. Check Submission Format"),
    code_cell("if os.path.exists('../submission.csv'):\n    sub_df = pd.read_csv('../submission.csv')\n    print(f'Submission shape: {sub_df.shape}')\n    print('\\nFirst 5 rows:')\n    print(sub_df.head())\nelse:\n    print('Submission.csv not generated yet.')")
]
create_notebook("notebooks/milestone_5_ens.ipynb", m5_cells)

# ----------------- KAGGLE COMBINED NOTEBOOK -----------------
kaggle_cells = [
    markdown_cell("# DL GenAI Project - Kaggle Inference Notebook (t22026)\n"
                  "This notebook is self-contained for submission to Kaggle. It implements context retrieval (RAG) and loaded deep learning models to predict the top 3 answers for each multiple choice prompt. It falls back gracefully to a robust TF-IDF similarity model if checkpoints are not available."),
    code_cell("import os\nimport re\nimport numpy as np\nimport pandas as pd\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import Dataset, DataLoader\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics.pairwise import cosine_similarity\n\n# Device setup\ndevice = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))\nprint(f'Running inference on: {device}')"),
    markdown_cell("## Data Preprocessing and Vocabulary (Model 1 Scratch components)"),
    code_cell("PAD_TOKEN = '<PAD>'\nUNK_TOKEN = '<UNK>'\nSEP_TOKEN = '<SEP>'\n\nclass MCQVocabulary:\n    def __init__(self, max_vocab_size=15000):\n        self.max_vocab_size = max_vocab_size\n        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2}\n        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SEP_TOKEN}\n        \n    def fit(self, texts):\n        word_counts = {}\n        for text in texts:\n            if not isinstance(text, str):\n                continue\n            cleaned = re.sub(r'[^\\w\\s]', '', text.lower())\n            for word in cleaned.split():\n                word_counts[word] = word_counts.get(word, 0) + 1\n        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)\n        for word, count in sorted_words[:self.max_vocab_size]:\n            if word not in self.word2idx:\n                idx = len(self.word2idx)\n                self.word2idx[word] = idx\n                self.idx2word[idx] = word\n                \n    def encode(self, text, max_len=128):\n        if not isinstance(text, str):\n            return [0] * max_len\n        cleaned = re.sub(r'[^\\w\\s]', '', text.lower())\n        tokens = cleaned.split()\n        encoded = [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in tokens]\n        if len(encoded) > max_len:\n            encoded = encoded[:max_len]\n        else:\n            encoded = encoded + [self.word2idx[PAD_TOKEN]] * (max_len - len(encoded))\n        return encoded\n\nclass ScratchMCQDataset(Dataset):\n    def __init__(self, df, vocab, max_len=128):\n        self.df = df.reset_index(drop=True)\n        self.vocab = vocab\n        self.max_len = max_len\n        \n    def __len__(self):\n        return len(self.df)\n        \n    def __getitem__(self, idx):\n        row = self.df.iloc[idx]\n        prompt = str(row['prompt'])\n        input_ids = []\n        for choice in ['A', 'B', 'C', 'D', 'E']:\n            choice_text = str(row[choice])\n            combined_text = f'{prompt} {SEP_TOKEN} {choice_text}'\n            encoded = self.vocab.encode(combined_text, max_len=self.max_len)\n            input_ids.append(encoded)\n        return torch.tensor(input_ids, dtype=torch.long)"),
    markdown_cell("## Model 1 Architecture: BiGRU + Attention"),
    code_cell("class SelfAttentionPooling(nn.Module):\n    def __init__(self, hidden_dim):\n        super().__init__()\n        self.attention = nn.Sequential(\n            nn.Linear(hidden_dim, hidden_dim // 2),\n            nn.Tanh(),\n            nn.Linear(hidden_dim // 2, 1)\n        )\n    def forward(self, rnn_outputs):\n        weights = self.attention(rnn_outputs)\n        weights = torch.softmax(weights, dim=1)\n        return torch.sum(rnn_outputs * weights, dim=1)\n\nclass BiGRUAttentionMCQModel(nn.Module):\n    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):\n        super().__init__()\n        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)\n        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)\n        self.attention = SelfAttentionPooling(hidden_dim * 2)\n        self.classifier = nn.Sequential(\n            nn.Linear(hidden_dim * 2, hidden_dim),\n            nn.ReLU(),\n            nn.Dropout(0.3),\n            nn.Linear(hidden_dim, 1)\n        )\n    def forward(self, input_ids):\n        batch_size, num_choices, seq_len = input_ids.shape\n        flat_input = input_ids.view(batch_size * num_choices, seq_len)\n        embedded = self.embedding(flat_input)\n        rnn_out, _ = self.gru(embedded)\n        pooled = self.attention(rnn_out)\n        logits = self.classifier(pooled)\n        return logits.view(batch_size, num_choices)"),
    markdown_cell("## Simple RAG Retriever"),
    code_cell("class SimpleVectorDB:\n    def __init__(self):\n        self.documents = []\n        self.vectorizer = TfidfVectorizer(stop_words='english')\n        self.doc_vectors = None\n    def fit(self, corpus):\n        self.documents = list(set([doc.strip() for doc in corpus if isinstance(doc, str) and len(doc.strip()) > 10]))\n        if self.documents:\n            self.doc_vectors = self.vectorizer.fit_transform(self.documents)\n    def query(self, text, k=2):\n        if not self.documents:\n            return []\n        query_vector = self.vectorizer.transform([text])\n        sims = cosine_similarity(query_vector, self.doc_vectors)[0]\n        top_indices = np.argsort(sims)[::-1][:k]\n        return [self.documents[idx] for idx in top_indices if sims[idx] > 0.0]"),
    markdown_cell("## Inference Execution (Graceful Fallback Implementation)"),
    code_cell("# Load test dataset\ntest_csv = 'data/test.csv' if os.path.exists('data/test.csv') else 'test.csv'\ntest_df = pd.read_csv(test_csv).fillna('')\n\ncheckpoint_scratch = 'checkpoints/scratch_model.pt'\ncheckpoint_pretrained = 'checkpoints/pretrained_model'\ncheckpoint_choice = 'checkpoints/choice_model'\n\nINDEX_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}\n\n# Flag to check if we can run DL ensembling\nrun_dl = os.path.exists(checkpoint_scratch) or os.path.exists(checkpoint_pretrained) or os.path.exists(checkpoint_choice)\n\nif run_dl:\n    print('Deep Learning models detected! Running model inference...')\n    # We initialize combined prediction arrays\n    num_samples = len(test_df)\n    combined_probs = np.zeros((num_samples, 5))\n    models_run = 0\n    \n    # 1. Model Scratch\n    if os.path.exists(checkpoint_scratch):\n        try:\n            print('Running Model 1 (Scratch BiGRU)...')\n            checkpoint = torch.load(checkpoint_scratch, map_location=device)\n            vocab = checkpoint['vocab']\n            model_1 = BiGRUAttentionMCQModel(vocab_size=checkpoint['vocab_size']).to(device)\n            model_1.load_state_dict(checkpoint['model_state_dict'])\n            model_1.eval()\n            \n            dataset_1 = ScratchMCQDataset(test_df, vocab)\n            loader_1 = DataLoader(dataset_1, batch_size=16, shuffle=False)\n            probs_scratch = []\n            with torch.no_grad():\n                for batch_x in loader_1:\n                    batch_x = batch_x.to(device)\n                    logits = model_1(batch_x)\n                    probs_scratch.append(torch.softmax(logits, dim=1).cpu().numpy())\n            combined_probs += 0.2 * np.concatenate(probs_scratch, axis=0)\n            models_run += 1\n        except Exception as e:\n            print('Error running Model 1:', e)\n            \n    # 2. Model Pretrained (DistilBERT)\n    if os.path.exists(checkpoint_pretrained):\n        try:\n            print('Running Model 2 (Pretrained DistilBERT)...')\n            from transformers import AutoTokenizer, AutoModelForMultipleChoice\n            tokenizer_2 = AutoTokenizer.from_pretrained(checkpoint_pretrained)\n            model_2 = AutoModelForMultipleChoice.from_pretrained(checkpoint_pretrained).to(device)\n            model_2.eval()\n            \n            probs_pretrained = []\n            for idx, row in test_df.iterrows():\n                prompt = str(row['prompt'])\n                first = [prompt] * 5\n                second = [str(row[opt]) for opt in ['A', 'B', 'C', 'D', 'E']]\n                inputs = tokenizer_2(first, second, truncation=True, max_length=128, padding='max_length', return_tensors='pt').to(device)\n                with torch.no_grad():\n                    outputs = model_2(input_ids=inputs['input_ids'].unsqueeze(0), attention_mask=inputs['attention_mask'].unsqueeze(0))\n                probs_pretrained.append(torch.softmax(outputs.logits, dim=1).cpu().numpy()[0])\n            combined_probs += 0.4 * np.array(probs_pretrained)\n            models_run += 1\n        except Exception as e:\n            print('Error running Model 2:', e)\n            \n    # 3. Model Choice (DeBERTa + LoRA + RAG)\n    if os.path.exists(checkpoint_choice):\n        try:\n            print('Running Model 3 (LoRA DeBERTa + RAG)...')\n            from transformers import AutoTokenizer, AutoModelForMultipleChoice\n            from peft import PeftModel\n            tokenizer_3 = AutoTokenizer.from_pretrained(checkpoint_choice)\n            base_model = AutoModelForMultipleChoice.from_pretrained('microsoft/deberta-v3-small')\n            model_3 = PeftModel.from_pretrained(base_model, checkpoint_choice).to(device)\n            model_3.eval()\n            \n            # Load RAG DB if training file is present\n            vector_db = SimpleVectorDB()\n            train_csv = 'data/train.csv' if os.path.exists('data/train.csv') else 'train.csv'\n            if os.path.exists(train_csv):\n                train_df = pd.read_csv(train_csv).fillna('')\n                corpus = []\n                for i, r in train_df.iterrows():\n                    corpus.append(r['prompt'])\n                    corpus.extend([r[c] for c in ['A', 'B', 'C', 'D', 'E']])\n                vector_db.fit(corpus)\n                \n            probs_choice = []\n            for idx, row in test_df.iterrows():\n                prompt = str(row['prompt'])\n                # Query context\n                contexts = vector_db.query(prompt, k=2)\n                filtered = [ctx for ctx in contexts if ctx.strip().lower() != prompt.strip().lower()][:2]\n                context_str = ' '.join(filtered)\n                aug_prompt = f'Context: {context_str}\\nQuestion: {prompt}' if filtered else prompt\n                \n                first = [aug_prompt] * 5\n                second = [str(row[opt]) for opt in ['A', 'B', 'C', 'D', 'E']]\n                inputs = tokenizer_3(first, second, truncation=True, max_length=192, padding='max_length', return_tensors='pt').to(device)\n                with torch.no_grad():\n                    outputs = model_3(input_ids=inputs['input_ids'].unsqueeze(0), attention_mask=inputs['attention_mask'].unsqueeze(0))\n                probs_choice.append(torch.softmax(outputs.logits, dim=1).cpu().numpy()[0])\n            combined_probs += 0.4 * np.array(probs_choice)\n            models_run += 1\n        except Exception as e:\n            print('Error running Model 3:', e)\n            \n    if models_run > 0:\n        # Generate Predictions\n        predictions = []\n        for probs in combined_probs:\n            sorted_indices = np.argsort(probs)[::-1][:3]\n            prediction_str = ' '.join([INDEX_MAP[idx] for idx in sorted_indices])\n            predictions.append(prediction_str)\n            \n        submission_df = pd.DataFrame({'id': test_df['id'], 'Prediction': predictions})\n    else:\n        run_dl = False\n        \nif not run_dl:\n    print('No model checkpoints detected or loaded. Falling back to TF-IDF semantic similarity...')\n    # Fallback to TF-IDF similarity to generate submission.csv without failure\n    predictions = []\n    for idx, row in test_df.iterrows():\n        prompt_cleaned = row['prompt'].lower()\n        choices = [str(row[c]).lower() for c in ['A', 'B', 'C', 'D', 'E']]\n        vectorizer = TfidfVectorizer()\n        try:\n            vectors = vectorizer.fit_transform([prompt_cleaned] + choices).toarray()\n            sims = cosine_similarity(vectors[0:1], vectors[1:])[0]\n        except:\n            sims = np.zeros(5)\n        choice_letters = ['A', 'B', 'C', 'D', 'E']\n        sorted_indices = np.argsort(sims)[::-1]\n        pred_str = ' '.join([choice_letters[i] for i in sorted_indices[:3]])\n        predictions.append(pred_str)\n        \n    submission_df = pd.DataFrame({'id': test_df['id'], 'Prediction': predictions})\n\n# Save submission file\nsubmission_df.to_csv('submission.csv', index=False)\nprint('Submission file saved successfully!')\nprint(submission_df.head())")
]
create_notebook("notebooks/DL-t22026-notebook.ipynb", kaggle_cells)
