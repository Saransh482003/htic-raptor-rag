import json
import os
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from sklearn.preprocessing import StandardScaler
from mechanics.chunk_clustering import gmm_umap_clustering, kmeans_umap_clustering, spectral_umap_clustering
from dotenv import load_dotenv
import warnings
import json
import json5
import re
warnings.filterwarnings("ignore")

load_dotenv()
import os

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Clustering and batching configuration (can be overridden via .env)
CLUSTERING_ALGO = os.getenv("CLUSTERING_ALGO", "gmm").lower()  # kmeans | gmm | spectral
UMAP_N_COMPONENTS = int(os.getenv("UMAP_N_COMPONENTS", 64))
UMAP_N_NEIGHBORS = int(os.getenv("UMAP_N_NEIGHBORS", 15))
CLUSTER_DIVISOR = int(os.getenv("CLUSTER_DIVISOR", 5))  # acts as divisor -> ceil(n_samples / divisor)
SUMMARY_BATCH_SIZE = int(os.getenv("SUMMARY_BATCH_SIZE", 5))
SUMMARY_BATCH_STRIDE = int(os.getenv("SUMMARY_BATCH_STRIDE", SUMMARY_BATCH_SIZE))  # < batch_size allows overlap
MAX_CHUNK_REUSE_PER_LEVEL = int(os.getenv("MAX_CHUNK_REUSE_PER_LEVEL", 2))  # allow limited overlap
MAX_CLUSTER_SIZE = int(os.getenv("MAX_CLUSTER_SIZE", 5))  # hard cap: partition clusters to size <= this

embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),base_url=OLLAMA_BASE_URL)
persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_store")
os.makedirs(persist_path, exist_ok=True)

vectorstore = Chroma(
    persist_directory=persist_path,
    embedding_function=embedding_model
)
collection = vectorstore._collection

def extract_clean_json(text):
    """Extract a JSON object with a 'summary' field from an LLM response.
    Guarantees a dict return of shape {"summary": <plain string>} without stringifying dicts.
    """
    def _clean_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    # If already a dict from upstream, coerce to expected shape
    if isinstance(text, dict):
        if isinstance(text.get("summary"), str):
            return {"summary": _clean_spaces(text["summary"])}
        # Try to salvage a string from any nested value
        def _first_str(v):
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                for vv in v.values():
                    s = _first_str(vv)
                    if s:
                        return s
            if isinstance(v, list):
                for vv in v:
                    s = _first_str(vv)
                    if s:
                        return s
            return None
        found = _first_str(text)
        return {"summary": _clean_spaces(found) if found else ""}

    # If not a string, stringify minimally but do not embed dict-looking text
    if not isinstance(text, str):
        return {"summary": _clean_spaces(str(text))}

    raw = text
    
    # 1) Try to locate a JSON-like block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = match.group(0) if match else None

    # Normalization helper for candidate JSON text
    def _normalize_jsonish(s: str) -> str:
        s = (
            s.replace("“", '"')
             .replace("”", '"')
             .replace("’", "'")
             .replace("`", "'")
             .replace("\u2011", "-")
             .replace("\u00A0", " ")
        )
        # Convert Python-style dicts to JSON
        if re.match(r"^\s*\{\'", s):
            s = re.sub(r"'(\w+)'(\s*):", r'"\1"\2:', s)
            def _replace_value_quotes(m):
                inner = m.group(1).replace('"', '\\"')
                return f': "{inner}"'
            s = re.sub(r":\s*'(.*?)'(?=[,\}])", _replace_value_quotes, s)
        # Keep escapes, but normalize newlines
        s = s.replace("\\n", " ")
        return s

    # 2) If we have a JSON-ish candidate, try to parse it strictly first
    if candidate:
        cand = _normalize_jsonish(candidate)
        for parser in (json.loads, json5.loads):
            try:
                parsed = parser(cand)
                if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
                    return {"summary": _clean_spaces(parsed["summary"])}
                # If parsed but no 'summary', try to extract a reasonable string
                def _first_str(v):
                    if isinstance(v, str):
                        return v
                    if isinstance(v, dict):
                        for vv in v.values():
                            s = _first_str(vv)
                            if s:
                                return s
                    if isinstance(v, list):
                        for vv in v:
                            s = _first_str(vv)
                            if s:
                                return s
                    return None
                found = _first_str(parsed)
                if found:
                    return {"summary": _clean_spaces(found)}
            except Exception:
                continue

        # 3) Regex-extract the summary field even if JSON is malformed
        m = re.search(r"[\"']summary[\"']\s*:\s*[\"'](.*?)[\"']\s*(?:[,}])", cand, re.DOTALL)
        if m:
            val = m.group(1)
            return {"summary": _clean_spaces(val)}

    # 4) Fallback: use the raw response text (not a JSON-looking string) as summary
    return {"summary": _clean_spaces(raw)}

def normalize_summary_text(text: str) -> str:
    """Normalize LLM summaries to be embedding-friendly and style-free.
    - remove markdown (**, __, ``, #, headings)
    - strip common section labels like "introduction:", "methods:", etc.
    - remove list bullets and extra punctuation spacing
    - fix spaced decimals (e.g., 2. 71 -> 2.71)
    - collapse whitespace; enforce lowercase
    """
    if not isinstance(text, str):
        text = str(text or "")

    t = text

    # Remove markdown emphasis and inline code markers
    t = t.replace("**", "").replace("__", "").replace("*", "")
    t = t.replace("`", "")
    # Remove markdown heading markers
    t = re.sub(r"^[ \t]*#+[ \t]*", "", t, flags=re.MULTILINE)

    # Remove common section labels while keeping content
    t = re.sub(r"\b(introduction|background|method|methods|materials and methods|results|discussion|conclusion|summary|objectives?|aims?)\s*:\s*",
               "", t, flags=re.IGNORECASE)

    # Strip leading bullets and dashes at line starts
    t = re.sub(r"^[ \t]*[\-•·\u2022]+[ \t]*", "", t, flags=re.MULTILINE)

    # Fix spaced decimals, avoid turning version numbers bad
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", t)

    # Tighten spaces near punctuation
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(\S)", r"\1 \2", t)

    # Collapse whitespace and newlines to a single space
    t = re.sub(r"\s+", " ", t).strip()

    # Enforce lowercase per requirement
    t = t.lower()
    return t

def create_summaries(batch, level):
    llm = ChatOllama(model=os.getenv("OLLAMA_SUMMARY_MODEL"), base_url=OLLAMA_BASE_URL,temperature=0)

    summary_prompt = PromptTemplate(
        input_variables=["text", "level"],
        template="""
            You are an expert medical technological summarizer.  
            Your task is to create a **concise but information-rich summary** of the following text, which has been chunked from a research paper or user manual or blog.  
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

            ### Most importantly:
            - The summary should start with the most critical and unique information first, prioritizing key findings, results, or conclusions over background details. Then other information can follow. Also avoid adding scientist names or citations who worked on the paper.

            ### Output format (STRICT):
            - Return ONLY strict JSON with a single key "summary".
            - The summary value must be: lowercased, plain sentences, no markdown, no headings or labels, no lists, no preambles (do not start with phrases like "as an expert..."), and no newlines.
            - Example:
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
    # Pass raw response; extractor returns {"summary": str}. Then normalize & lowercase strictly.
    summary = extract_clean_json(response)
    summary["summary"] = normalize_summary_text(summary["summary"])
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

            # Ensure ALL chunks at this level are embedded (handle partials)
            results = collection.get(
                where={"id": {"$in": chunk_ids}},
                include=["embeddings", "documents", "metadatas"]
            )
            present_meta = results.get("metadatas") or []
            present_ids = set(m.get("id") for m in present_meta if isinstance(m, dict))
            missing_chunks = [ch for ch in current_level if ch["id"] not in present_ids]

            if missing_chunks:
                docs = []
                for chunk in missing_chunks:
                    docs.append({
                        "id": chunk["id"],
                        "text": chunk.get("text", ""),
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

                # Re-fetch to include newly added embeddings
                results = collection.get(
                    where={"id": {"$in": chunk_ids}},
                    include=["embeddings", "documents", "metadatas"]
                )

            if len(current_level) > 5:
                embeddings = np.array(results["embeddings"])
                std_embeddings = StandardScaler().fit_transform(embeddings)
                # Select clustering algorithm
                if CLUSTERING_ALGO == "gmm":
                    umap_embeddings, cluster_labels = gmm_umap_clustering(
                        std_embeddings,
                        n_components=UMAP_N_COMPONENTS,
                        n_neighbors=UMAP_N_NEIGHBORS,
                        n_clusters=CLUSTER_DIVISOR,
                    )
                elif CLUSTERING_ALGO == "spectral":
                    umap_embeddings, cluster_labels = spectral_umap_clustering(
                        std_embeddings,
                        n_components=UMAP_N_COMPONENTS,
                        n_neighbors=UMAP_N_NEIGHBORS,
                        n_clusters=CLUSTER_DIVISOR,
                    )
                else:
                    umap_embeddings, cluster_labels = kmeans_umap_clustering(
                        std_embeddings,
                        n_components=UMAP_N_COMPONENTS,
                        n_neighbors=UMAP_N_NEIGHBORS,
                        n_clusters=CLUSTER_DIVISOR,
                    )

                clusters = np.unique(cluster_labels)
                text_chunks = np.array([chunk for chunk in current_level])
                # Track reuse per chunk to cap overlap per level
                reuse_counts = {chunk["id"]: 0 for chunk in current_level}

                for i in clusters:
                    cluster_mask = (cluster_labels == i)
                    cluster_chunks = text_chunks[cluster_mask]
                    total = len(cluster_chunks)
                    if total == 0:
                        continue

                    # Partition clusters larger than MAX_CLUSTER_SIZE
                    parts = [list(cluster_chunks[k:k + MAX_CLUSTER_SIZE]) for k in range(0, total, MAX_CLUSTER_SIZE)]
                    print(f"\t🔹 Cluster {i}: size={total}, partitions={len(parts)} (max {MAX_CLUSTER_SIZE} per partition)")

                    for p_idx, part in enumerate(parts):
                        if not part:
                            continue
                        if len(part) == 1:
                            summary = create_summaries(part, next_level_num)
                            next_level.append({
                                "id": f"{file}_summary_{i}_part_{p_idx}_level_{next_level_num}",
                                "text": summary["summary"],
                                "source": [part[0]["id"]]
                            })
                            # Update reuse counts
                            reuse_counts[part[0]["id"]] = reuse_counts.get(part[0]["id"], 0) + 1
                            continue

                        # Enforce per-level reuse cap within partition (typically used once)
                        filtered = []
                        for ch in part:
                            cid = ch["id"]
                            if reuse_counts.get(cid, 0) < MAX_CHUNK_REUSE_PER_LEVEL:
                                filtered.append(ch)

                        if not filtered:
                            continue

                        summary = create_summaries(filtered, next_level_num)
                        next_level.append({
                            "id": f"{file}_summary_{i}_part_{p_idx}_level_{next_level_num}",
                            "text": summary["summary"],
                            "source": [c["id"] for c in filtered]
                        })
                        # Update reuse counts
                        for c in filtered:
                            reuse_counts[c["id"]] = reuse_counts.get(c["id"], 0) + 1
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