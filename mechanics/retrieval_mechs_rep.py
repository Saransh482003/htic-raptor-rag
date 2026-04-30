import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os

load_dotenv()

# Load vectorstore
embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"), base_url="http://127.0.0.1:11434")
print(os.getenv("CHROMA_PERSIST_PATH", "./chromaDB_store"))
vectorstore = Chroma(
    persist_directory=os.getenv("CHROMA_PERSIST_PATH", "./chromaDB_store"),
    embedding_function=embeddings
)
# Load original summary tree (for hierarchy reference if needed)
with open(os.getenv("HIERARCHY_STORE_PATH"), "r", encoding="utf-16") as f:
    summary_tree = json.load(f)

def raptor_retrieve(query, summary_tree, top_k_root=1, top_k_children=2):
    """
    Perform RAPTOR-style hierarchical retrieval.
    Returns all documents from all levels traversed (root summaries, intermediate summaries, and leaf chunks).
    """

    # Helper to append only unseen documents (by stable chunk id)
    query = query.lower()

    def add_unique(results, collector, seen_ids):
        for doc in results or []:
            doc_id = doc.metadata.get("id") if hasattr(doc, "metadata") else None
            # Fallback to a hash of content if id is missing (very rare)
            if not doc_id:
                doc_id = f"pc:{hash(getattr(doc, 'page_content', ''))}"
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            collector.append(doc)

    # Collect root (max-level) chunk IDs and their level keys per file
    root_ids = []
    max_levels = set()
    for file_id, file_data in summary_tree.items():
        levels = list(file_data["levels"].keys())
        if not levels:
            continue
        max_level_key = max(levels, key=lambda x: int(x.split("_")[1]))
        max_levels.add(max_level_key)
        for chunk in file_data["levels"][max_level_key]:
            root_ids.append(chunk["id"])

    # Try ID-filtered root search first
    root_results = []
    if root_ids:
        root_results = vectorstore.similarity_search(
            query,
            k=min(top_k_root, len(root_ids)),
            filter={"id": {"$in": root_ids}}
        )

    # Fallback: filter by level if ID filter returns nothing (some setups block filtering by 'id')
    if not root_results and max_levels:
        root_results = vectorstore.similarity_search(
            query,
            k=top_k_root,
            filter={"level": {"$in": list(max_levels)}}
        )

    # Store all retrieved documents from all levels
    all_results = []
    seen_ids = set()
    add_unique(root_results, all_results, seen_ids)  # Add root-level summaries uniquely

    def descend(children_ids):
        if not children_ids:
            return

        # Retrieve next-level candidates among provided children IDs
        # De-duplicate incoming children ids to avoid repeated fetches
        unique_children_ids = list(dict.fromkeys(children_ids))
        k = min(top_k_children, len(unique_children_ids))
        child_results = vectorstore.similarity_search(
            query,
            k=k,
            filter={"id": {"$in": unique_children_ids}}
        )
        add_unique(child_results, all_results, seen_ids)  # Add intermediate summaries uniquely

        # Build next generation from child chunk_source metadata
        next_children_ids = []
        for doc in child_results:
            src_raw = doc.metadata.get("chunk_source")
            try:
                src_list = json.loads(src_raw) if isinstance(src_raw, str) else (src_raw or [])
            except Exception:
                src_list = []
            if isinstance(src_list, list):
                next_children_ids.extend(src_list)

        # If there are no next children, these are leaves; include ALL leaves (not just top-k)
        if not next_children_ids:
            if k < len(unique_children_ids):
                leaf_all = vectorstore.similarity_search(
                    query,
                    k=len(unique_children_ids),
                    filter={"id": {"$in": unique_children_ids}}
                )
                add_unique(leaf_all, all_results, seen_ids)
            return

        # Continue traversal
        # Dedupe next-generation children to prevent repeated descent
        descend(list(dict.fromkeys(next_children_ids)))

    children = []
    for doc in root_results:
        src_raw = doc.metadata.get("chunk_source")
        try:
            src_list = json.loads(src_raw) if isinstance(src_raw, str) else (src_raw or [])
        except Exception:
            src_list = []
        if isinstance(src_list, list):
            children.extend(src_list)
    # De-duplicate initial children before descent
    descend(list(dict.fromkeys(children)))
    return all_results

import json
with open("./essentials/summary_tree.json", "r", encoding="utf-16") as f:
    summary_tr = json.loads(f.read())
response = raptor_retrieve("What is Sphygmocor?",summary_tr)