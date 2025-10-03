import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load summary tree
with open("./essentials/summary_tree.json", "r", encoding="utf-16") as f:
    data = json.load(f)

# Flatten chunks + summaries into docs
docs = []
for file_id, file_data in data.items():
    for level_name, chunks in file_data["levels"].items():
        level_num = int(level_name.split("_")[1])
        for chunk in chunks:
            docs.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": {
                    "id": chunk["id"],          # needed for filtering later
                    "file_id": file_id,
                    "level": level_num,
                    "chunk_source": json.dumps(chunk.get("source", []))
                }
            })

# Prepare texts and metadatas
texts = [d["text"] for d in docs]
metadatas = [d["metadata"] for d in docs]

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize Chroma vectorstore (persistent)
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_store"  # where data will be saved
)

vectorstore.persist()  # ensure it’s saved to disk
print("✅ Chroma index built and saved to ./chroma_store")
