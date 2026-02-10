import requests
import os
import io

SAMPLE_DOCS = {}

API_URL = "http://127.0.0.1:8000"


def load_sample_files():
    sample_dir = "sample_texts"
    for filename in os.listdir(sample_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(sample_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                SAMPLE_DOCS[filename] = f.read()
    print(f"loaded {len(SAMPLE_DOCS)} files from {sample_dir}/")


def upload_document(user_id: str, filename: str, content: str):
    response = requests.post(
        f"{API_URL}/upload",
        data={"user_id": user_id, "document_name": filename},
        files={"file": (filename, io.BytesIO(
            content.encode("utf-8")), "text/plain")}
    )
    if response.status_code == 200:
        print(f"{filename} uploaded successfully")
    else:
        print(f"{filename} upload failed: {response.text}")


def preprocess_samples():
    test_user_id = "test_user_001"
    for filename, content in SAMPLE_DOCS.items():
        print(f"processing {filename}")
        upload_document(test_user_id, filename, content)
    print("preprocessing complete")


if __name__ == "__main__":
    load_sample_files()
    preprocess_samples()
