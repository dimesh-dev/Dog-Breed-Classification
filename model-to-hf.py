import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo

# Load environment variables
load_dotenv()

token = os.getenv("HF_TOKEN")
HfFolder.save_token(token)

repo_id = "dimeshanthoney/dog-breed-classifier"
api = HfApi()

# Create repo if it doesn’t exist
try:
    api.create_repo(repo_id=repo_id, repo_type="model", token=token)
    print("✅ Repo created.")
except Exception as e:
    print(f"⚠️ Repo may already exist: {e}")

# Upload the model file
upload_file(
    path_or_fileobj="Image_classify.keras",
    path_in_repo="Image_classify.keras",
    repo_id=repo_id,
    repo_type="model",
    token=token
)

print("✅ Model uploaded to Hugging Face Hub!")
