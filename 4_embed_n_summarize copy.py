import json
import os
import numpy as np
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from sklearn.preprocessing import StandardScaler
from mechanics.chunk_clustering import gmm_umap_clustering
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import warnings
import json
import json5
import re
warnings.filterwarnings("ignore")

load_dotenv()
import os

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),base_url=OLLAMA_BASE_URL)
persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_store")
os.makedirs(persist_path, exist_ok=True)

vectorstore = Chroma(
    persist_directory=persist_path,
    embedding_function=embedding_model
)
collection = vectorstore._collection

def extract_clean_json(text: str):
    """Safely extract and clean JSON-like data (even if single-quoted or malformed)."""
    if not isinstance(text, str):
        return text  # already parsed
    
    # 1️⃣ Extract JSON-like portion
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"summary": text.strip()}
    candidate = match.group(0)
    
    # 2️⃣ Normalize quotes and weird Unicode
    candidate = (
        candidate.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("`", "'")
        .replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u00A0", " ")  # non-breaking space
    )
    
    # 3️⃣ Detect Python-style dict → convert to JSON
    if re.match(r"^\s*\{\'", candidate):  
        # Replace single quotes around keys
        candidate = re.sub(r"'(\w+)'(\s*):", r'"\1"\2:', candidate)
        # Replace single quotes around string values safely
        def _replace_value_quotes(m):
            inner = m.group(1).replace('"', '\\"')
            return f': "{inner}"'
        candidate = re.sub(r":\s*'(.*?)'(?=[,\}])", _replace_value_quotes, candidate)
    
    # 4️⃣ Remove extra backslashes or newlines
    candidate = candidate.replace("\\n", " ").replace("\\", "")
    
    # 5️⃣ Try parsing
    for parser in (json.loads, json5.loads):
        try:
            return parser(candidate)
        except Exception:
            continue
    
    # 6️⃣ Fallback: return text safely wrapped in JSON
    return {"summary": candidate.strip()}

def create_summaries(batch, level):
    llm = ChatOllama(model=os.getenv("OLLAMA_SUMMARY_MODEL"), base_url=OLLAMA_BASE_URL,temperature=0)

    summary_prompt = PromptTemplate(
        input_variables=["text", "level"],
        template="""
            You are an expert scientific summarizer for a retrieval-augmented generation (RAG) system.  
            Your task is to create a **concise but information-rich summary** of the following text.  
            The summary will later be used recursively to build a hierarchical knowledge tree (RAPTOR), so it must be coherent, self-contained, and faithful.

            ### Guidelines:
            - **Faithfulness**: Do NOT introduce facts not present in the input text. Summarize only what is given.  
            - **Coverage**: Capture the most important entities, concepts, and relationships mentioned in the text.  
            - **Abstraction**: Shorten long explanations while preserving key details (e.g., thresholds, conditions, findings).  
            - **Clarity**: Write in clear, concise prose; avoid repetition, filler, or references.  
            - **Specificity**: Retain critical domain-specific terms (e.g., "electrocardiogram-based analysis", "≥ 5 events per hour").  
            - **Context Independence**: The summary should stand on its own, without requiring the reader to see the original text.  
            - **Length Control**: 
            - For lower-level chunks (level 0-1), produce ~3-5 sentences.  
            - For higher-level summaries (level ≥ 2), focus more on abstraction and generalization, keeping it 2-3 sentences.  

            ### Output format:
            Return the summary in **strict JSON** format only, no explanations, no extra text, no invalid punctuations. A single key "summary". 
            The response should be plain simple text, NO markdown styling, no new line charaters
            Example:
            {{
                "summary": "Concise summary text here."
            }}

            ### Input text (level {level}):
            <<<{text}>>>
        """
    )

    text = "\n\n".join([chunk["text"] for chunk in batch])
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    response = summary_chain.run(text=text, level=level)
    summary = extract_clean_json(response[response.find('{'):response.rfind('}')+1].replace("\\", "/").replace("\"", "\'").replace("\n", " "))
    return summary



with open(os.getenv("HIERARCHY_STORE_PATH"), 'r', encoding="utf-16") as f:
    summary_tree = json.load(f)

for file in summary_tree.keys():
    print(f"Processing {file}...")

    summary_levels = summary_tree[file]["levels"]
    max_level = list(summary_levels.keys())[-1]

    if len(summary_levels[max_level]) == 1:
        print(f"🟣 Final summary level detected for {file} ({max_level})")

        current_level = summary_levels[max_level]
        chunk_ids = [chunk["id"] for chunk in current_level]

        # Check if already embedded
        results = collection.get(
            where={"id": {"$in": chunk_ids}},
            include=["embeddings", "documents", "metadatas"]
        )

        embeddings = np.array(results["embeddings"])
        if embeddings.size == 0:
            print(f"\t⚙️ Embedding final level summary (ID: {chunk_ids[0]})")

            docs = []
            for chunk in current_level:
                text = chunk.get("text", "").strip()
                if not text:
                    print(f"\t⚠️ No text found for {chunk['id']}, skipping.")
                    continue

                docs.append({
                    "id": chunk["id"],
                    "text": text,
                    "metadata": {
                        "id": chunk["id"],
                        "file_id": file,
                        "level": max_level,
                        "chunk_source": json.dumps(chunk.get("source", []))
                    }
                })

            if docs:
                texts = [d["text"] for d in docs]
                metadatas = [d["metadata"] for d in docs]
                vectorstore.add_texts(texts=texts, metadatas=metadatas)
                print(f"\t✅ Final summary embedded successfully ({len(docs)} doc).")
        else:
            print("\t✅ Final summary already embedded.")

    else:
        while len(summary_levels[max_level]) > 1:
            print("Current Level:", max_level)
            print("Number of chunks at this level:", len(summary_levels[max_level]))

            current_level = summary_levels[max_level]
            next_level = []
            next_level_num = int(max_level.split('_')[1]) + 1
            next_name = f"level_{next_level_num}"
            chunk_ids = [chunk["id"] for chunk in current_level]

            print(f"\t🟠 Starting Level: {next_name}...")

            results = collection.get(
                where={"id": {"$in": chunk_ids}},
                include=["embeddings", "documents", "metadatas"]
            )
            embeddings = np.array(results["embeddings"])
            if embeddings.size == 0:
                docs = []
                for chunk in current_level:
                    docs.append({
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": {
                            "id": chunk["id"],
                            "file_id": file,
                            "level": max_level,
                            "chunk_source": json.dumps(chunk.get("source", []))
                            }
                        })

                texts = [d["text"] for d in docs]
                metadatas = [d["metadata"] for d in docs]
                vectorstore.add_texts(texts=texts, metadatas=metadatas)

            results = collection.get(
                where={"id": {"$in": chunk_ids}},
                include=["embeddings", "documents", "metadatas"]
            )

            if len(current_level) > 5:
                embeddings = np.array(results["embeddings"])
                std_embeddings = StandardScaler().fit_transform(embeddings)
                umap_embeddings, cluster_labels = gmm_umap_clustering(std_embeddings, n_components=int(os.getenv("UMAP_N_COMPONENTS", 64)), n_neighbors=int(os.getenv("UMAP_N_NEIGHBORS", 15)))

                clusters = np.unique(cluster_labels)
                text_chunks = np.array([chunk for chunk in current_level])
                for i in clusters:
                    cluster_chunks = text_chunks[cluster_labels == i]
                    for j in range(0, len(cluster_chunks), 5):
                        batch = cluster_chunks[j:j + 5]
                        summary = create_summaries(batch, next_level_num)
                        next_level.append({
                            "id": f"{file}_summary_{i}_{j//5}_level_{next_level_num}",
                            "text": summary["summary"],
                            "source": [chunk["id"] for chunk in batch]
                        })
                    print(f"\t🔹 Summarized cluster {i} batch {j}-{j+4} -> Summary ID: {file}_summary_{i}_{j//5}_level_{next_level_num}")
            else:
                text_chunks = np.array([chunk for chunk in current_level])
                summary = create_summaries(text_chunks, next_level_num)
                next_level.append({
                    "id": f"{file}_summary_{0}_{0}_level_{next_level_num}",
                    "text": summary["summary"],
                    "source": [chunk["id"] for chunk in text_chunks]
                })
                print(f"\t🔹 Summarized final cluster {0} -> Summary ID: {file}_summary_{0}_{0}_level_{next_level_num}")
            summary_levels[next_name] = next_level
            summary_tree[file]["levels"] = summary_levels   
            max_level = next_name

            print(f"\t🟢 Completed Level {next_level_num}; Total Chunks: {len(next_level)}")

            with open(os.getenv("HIERARCHY_STORE_PATH"), 'w', encoding="utf-16") as out_f:
                json.dump(summary_tree, out_f, ensure_ascii=False, indent=4)
        print(f"✅ Finished processing {file}.\n")