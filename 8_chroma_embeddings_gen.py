import os
import json
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# ========================
# Load environment variables
# ========================
load_dotenv()

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_store")
summary_path = os.getenv("HIERARCHY_STORE_PATH", "./summary_tree.json")

# ========================
# Recreate fresh Chroma DB
# ========================
if os.path.exists(persist_path):
    import shutil
    print(f"🧹 Removing old Chroma database at {persist_path} ...")
    shutil.rmtree(persist_path)

os.makedirs(persist_path, exist_ok=True)
print(f"✅ Created clean Chroma directory: {persist_path}")

# ========================
# Initialize embedding model
# ========================
embedding_model = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),
    base_url=OLLAMA_BASE_URL
)

vectorstore = Chroma(
    persist_directory=persist_path,
    embedding_function=embedding_model
)

# ========================
# Load your hierarchy summaries
# ========================
with open(summary_path, 'r', encoding="utf-16") as f:
    summary_tree = json.load(f)

# ========================
# Embed summaries into Chroma
# ========================
total_docs = 0

for file_name, file_data in summary_tree.items():
    print(f"\n📘 Processing file: {file_name}")
    summary_levels = file_data.get("levels", {})

    for level_name, chunks in summary_levels.items():
        print(f"  ↳ Embedding {len(chunks)} chunks from {level_name} ...")

        texts, metadatas = [], []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            texts.append(text)
            metadatas.append({
                "id": chunk.get("id", ""),
                "file_id": file_name,
                "level": level_name,
                "chunk_source": json.dumps(chunk.get("source", []))
            })

        if texts:
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            total_docs += len(texts)

print(f"\n✅ Re-embedding completed successfully.")
print(f"📊 Total documents embedded: {total_docs}")
print(f"💾 Chroma persisted at: {persist_path}")
