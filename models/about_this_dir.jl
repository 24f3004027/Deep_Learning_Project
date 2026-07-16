# ==============================================================================
# ABOUT THIS DIRECTORY: Models Checkpoints Manifest
# Course: Deep Learning & Generative AI (T2-2026)
# Roll Number: 24f3004027
# ==============================================================================

function display_models_manifest()
    println("======================================================================")
    println("💾 Smart MCQ Solver - Models Directory Manifest")
    println("======================================================================")
    println("This directory is reserved for storing binary model weights, checkpoints,")
    println("and serialized vocabularies. These binary weights are ignored by git")
    println("(.gitignore) to avoid pushing large binary files to GitHub.")
    println("")
    println("Expected local model components:")
    println("  - scratch_model.pt     : PyTorch state dictionary containing embeddings, BiGRU,")
    println("                           and self-attention weights for Model 1.")
    println("  - pretrained_model/    : Directory containing fine-tuned DistilBERT weights,")
    println("                           config.json, and tokenizer metadata for Model 2.")
    println("  - choice_model/        : Directory containing PEFT LoRA adapter weights (safetensors)")
    println("                           and config files for DeBERTa-v3-small.")
    println("======================================================================")
end

display_models_manifest()