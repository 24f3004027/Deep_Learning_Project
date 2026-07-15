import os
import argparse
from huggingface_hub import HfApi, login

def deploy_to_hf_space(repo_name="DL-t22026-MCQ-App", token=None):
    api = HfApi()
    
    # 1. Login to Hugging Face
    if token:
        print("Logging in to Hugging Face with provided token...")
        login(token=token, write_permission=True)
    else:
        print("Checking Hugging Face credentials...")
        # Will prompt or check existing git/cache credentials
        login(write_permission=True)
        
    # Get current user namespace
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating Hugging Face Space: '{repo_id}' with Gradio SDK...")
    try:
        # Create Space repo
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print("Space repository created or verified successfully!")
    except Exception as e:
        print(f"Error creating space repository: {e}")
        return
        
    # 2. Files to upload
    files_to_upload = {
        "app.py": "app.py",
        "requirements.txt": "requirements.txt",
    }
    
    # Check if scratch model checkpoint exists to deploy it too!
    scratch_model_path = "checkpoints/scratch_model.pt"
    if os.path.exists(scratch_model_path):
        # We upload it under checkpoints directory in the Space
        files_to_upload[scratch_model_path] = "checkpoints/scratch_model.pt"
        print(f"Detected trained scratch model checkpoint: '{scratch_model_path}'. It will be deployed to the Space.")
    else:
        print("Warning: No scratch model checkpoint found. App will fallback to TF-IDF similarity on the Space.")
        
    print("\nUploading files to Hugging Face Space...")
    for local_path, repo_path in files_to_upload.items():
        if os.path.exists(local_path):
            print(f"Uploading '{local_path}' to '{repo_path}'...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="space"
                )
                print(f"Successfully uploaded {local_path}!")
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")
        else:
            print(f"Warning: File {local_path} does not exist. Skipping.")
            
    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDeployment Complete! View your space at: {space_url}")
    print("Add this URL to your Form 3 (Deployment Link) submission.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, default="DL-t22026-MCQ-App", help="Hugging Face Space repository name")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (optional if already logged in)")
    args = parser.parse_args()
    
    deploy_to_hf_space(args.repo_name, args.token)
