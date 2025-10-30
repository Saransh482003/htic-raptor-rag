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
persist_dir = os.getenv("CHROMA_PERSIST_PATH", "./chroma_store")
print(persist_dir)
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)
# Load original summary tree (for hierarchy reference if needed)
with open(os.getenv("HIERARCHY_STORE_PATH"), "r", encoding="utf-16") as f:
    summary_tree = json.load(f)

def raptor_retrieve(query, summary_tree, top_k_root=1, top_k_children=2, return_by_level=False):
    """
    Perform RAPTOR-style hierarchical retrieval.
    Returns all documents from all levels traversed (root summaries, intermediate summaries, and leaf chunks).

    Branching semantics:
    - Pick top_k_root roots globally across all files (max-level summaries).
    - For each selected parent on a level, pick up to top_k_children of ITS children by similarity.
    - Continue level-by-level until leaves (or no children), collecting every node visited.

    This yields up to: top_k_root + top_k_root*top_k_children + top_k_root*top_k_children^2 + ... per depth.
    """

    # Collect root (max-level) chunk IDs and their level keys per file
    query = query.lower()
    root_ids = []
    max_levels = set()
    for _, file_data in summary_tree.items():
        levels = list(file_data.get("levels", {}).keys())
        if not levels:
            continue
        max_level_key = max(levels, key=lambda x: int(x.split("_")[1]))
        max_levels.add(max_level_key)
        for chunk in file_data["levels"][max_level_key]:
            root_ids.append(chunk["id"])

    # Try ID-filtered root search first (restrict to max-level summaries only)
    root_results = []
    if root_ids:
        root_scored = vectorstore.similarity_search_with_score(
            query,
            k=min(top_k_root, len(root_ids)),
            filter={"id": {"$in": root_ids}},
        )
        root_results = [d for d, _ in root_scored]
        # attach scores for traceability
        for d, s in root_scored:
            try:
                d.metadata["score"] = float(s)
            except Exception:
                pass

    # Fallback: filter by level if ID filter returns nothing (some setups block filtering by 'id')
    if not root_results and max_levels:
        root_scored = vectorstore.similarity_search_with_score(
            query,
            k=top_k_root,
            filter={"level": {"$in": list(max_levels)}},
        )
        root_results = [d for d, _ in root_scored]
        for d, s in root_scored:
            try:
                d.metadata["score"] = float(s)
            except Exception:
                pass

    # Store all retrieved documents from all levels
    all_results = []
    by_level = []  # list[list[Document]] collected per depth (0 = roots)
    seen_ids = set()

    def _append(doc):
        mid = doc.metadata.get("id")
        if mid and mid in seen_ids:
            return
        if mid:
            seen_ids.add(mid)
        all_results.append(doc)

    # Always include the selected root summaries in the return
    for d in root_results:
        _append(d)
    if root_results:
        by_level.append(list(root_results))

    # Helper to parse children list from a document's metadata
    def _children_ids_of(doc):
        src_raw = doc.metadata.get("chunk_source")
        try:
            src_list = json.loads(src_raw) if isinstance(src_raw, str) else (src_raw or [])
        except Exception:
            src_list = []
        return src_list if isinstance(src_list, list) else []

    # Level-order traversal with per-parent top-k child selection
    current_level_parents = list(root_results)
    while current_level_parents:
        next_level_parents = []
        for parent in current_level_parents:
            child_ids = _children_ids_of(parent)
            if not child_ids:
                continue

            k = min(top_k_children, len(child_ids))
            try:
                child_scored = vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter={"id": {"$in": child_ids}},
                )
            except Exception:
                child_scored = []

            child_results = [d for d, _ in child_scored]
            for d, s in child_scored:
                try:
                    d.metadata["score"] = float(s)
                except Exception:
                    pass

            # Add chosen children to results and queue for the next level
            for d in child_results:
                _append(d)
            next_level_parents.extend(child_results)

        # Advance to next level
        if next_level_parents:
            by_level.append(list(next_level_parents))
        current_level_parents = next_level_parents

    if return_by_level:
        return {"all": all_results, "levels": by_level}
    return all_results

import json
with open("./essentials/summary_tree.json", "r", encoding="utf-16") as f:
    summary_tr = json.loads(f.read())

out = raptor_retrieve("What is ARTSENS?", summary_tr, top_k_root=1, top_k_children=2, return_by_level=True)
results = out["all"]
levels = out["levels"]
print(f"Retrieved {len(results)} documents across {len(levels)} levels.")
print("Counts by level:", [len(l) for l in levels])
print("\nShowing metadata and first ~100 words of each:\n")

def _first_n_words(text: str, n: int = 100) -> str:
    words = (text or "").split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + " …"

for idx, doc in enumerate(results, 1):
    meta = getattr(doc, "metadata", {}) or {}
    # Try to get the textual content
    try:
        content = doc.page_content
    except Exception:
        content = str(getattr(doc, "document", ""))

    # Print compact metadata line
    print(f"[{idx}] id={meta.get('id')} file={meta.get('file_id')} level={meta.get('level')} score={meta.get('score')}")
    # Print snippet up to ~100 words
    print(_first_n_words(content, 100))
    print()