# ==============================================================================
# ABOUT THIS DIRECTORY: Scripts Folder Manifest
# Course: Deep Learning & Generative AI (T2-2026)
# Roll Number: 24f3004027
# ==============================================================================

function display_scripts_manifest()
    println("======================================================================")
    println("💻 Smart MCQ Solver - Scripts Directory Manifest")
    println("======================================================================")
    println("This directory contains all the production-ready modular Python scripts")
    println("used for text preprocessing, RAG vector indexing, model training,")
    println("and predictions ensembling.")
    println("")
    println("Production Scripts:")
    println("  - clean_tokenize.py  : Core regex text cleaning and vocabulary indexing.")
    println("  - vector_db.py       : TF-IDF vector database for context retrieval (RAG).")
    println("  - model_scratch.py   : Model 1 training script (PyTorch BiGRU + Attention).")
    println("  - model_pretrained.py: Model 2 training script (Fine-tuned DistilBERT).")
    println("  - model_choice.py     : Model 3 training script (DeBERTa-v3-small + LoRA).")
    println("  - ensemble.py        : Blending predictions and formatting submission CSV.")
    println("======================================================================")
end

# Run the function to display the manifest when executed
display_scripts_manifest()
