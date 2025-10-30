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

    root_level_map = {}
    for file_id, file_data in summary_tree.items():
        levels = list(file_data["levels"].keys())
        max_level = max(levels, key=lambda x: int(x.split("_")[1]))
        for chunk in file_data["levels"][max_level]:
            root_level_map[chunk["id"]] = int(max_level.split("_")[1])
    root_ids = list(root_level_map.keys())
    print(root_ids)
    # Retrieve top-k root-level summaries
    root_results = vectorstore.similarity_search(
        query,
        k=top_k_root,
        filter={"id": {"$in": root_ids}}
    )
    print("root_ids", root_results)

    # Store all retrieved documents from all levels
    all_results = []
    all_results.extend(root_results)  # Add root-level summaries

    def descend(children):
        if all(cid.endswith("level_0") for cid in children):
            child_results = vectorstore.similarity_search(
                query,
                k=len(children),
                filter={"metadata.id": {"$in": children}}
            )
            all_results.extend(child_results)  # Add leaf chunks
            return
        
        child_results = vectorstore.similarity_search(
            query,
            k=min(top_k_children, len(children)),
            filter={"metadata.id": {"$in": children}}
        )
        all_results.extend(child_results)  # Add intermediate summaries
        
        next_children = []
        for doc in child_results:
            next_children.extend(json.loads(doc.metadata["chunk_source"]))
        descend(next_children)

    children = []
    for doc in root_results:
        children.extend(json.loads(doc.metadata["chunk_source"]))
    descend(children)
    print(all_results)
    return all_results

# ...existing imports and setup...

# def raptor_retrieve(query, summary_tree, top_k_root=2, top_k_children=3):
#     """
#     Perform RAPTOR-style hierarchical retrieval.
#     Returns all documents from all levels traversed (root summaries, intermediate summaries, and leaf chunks).
#     """

#     # 1) Build list of root (top-level) summary IDs per file
#     root_ids = []
#     for file_id, file_data in summary_tree.items():
#         levels = list(file_data["levels"].keys())
#         max_level_key = max(levels, key=lambda x: int(x.split("_")[1]))
#         for chunk in file_data["levels"][max_level_key]:
#             root_ids.append(chunk["id"])

#     # Helper: de-dup by metadata.id
#     seen_ids = set()
#     def add_unique(results, sink):
#         for d in results:
#             did = d.metadata.get("id")
#             if did and did not in seen_ids:
#                 seen_ids.add(did)
#                 sink.append(d)

#     # 2) Retrieve top-k root summaries (FIX: filter on metadata key "id", not "metadata.id")
#     root_results = vectorstore.similarity_search(
#         query,
#         k=min(top_k_root, len(root_ids)) if root_ids else 0,
#         filter={"id": {"$in": root_ids}}
#     )

#     all_results = []
#     add_unique(root_results, all_results)

#     # 3) Recursively descend using chunk_source; detect leaves via empty chunk_source
#     def descend(children_ids):
#         if not children_ids:
#             return

#         # Retrieve a focused subset at the current level
#         k = min(top_k_children, len(children_ids))
#         child_results = vectorstore.similarity_search(
#             query,
#             k=k,
#             filter={"id": {"$in": children_ids}}
#         )
#         add_unique(child_results, all_results)

#         # Compute next generation from the subset
#         next_children_ids = []
#         for doc in child_results:
#             src = doc.metadata.get("chunk_source", "[]")
#             try:
#                 next_children_ids.extend(json.loads(src))
#             except Exception:
#                 # If malformed, treat as leaf
#                 pass

#         # If no next children -> we are at leaves.
#         # Ensure we include ALL leaves, not just the top_k subset.
#         if not next_children_ids:
#             if k < len(children_ids):
#                 leaf_all = vectorstore.similarity_search(
#                     query,
#                     k=len(children_ids),
#                     filter={"id": {"$in": children_ids}}
#                 )
#                 add_unique(leaf_all, all_results)
#             return

#         # Continue down
#         descend(next_children_ids)

#     # Seed descent from root results
#     seed_children = []
#     for doc in root_results:
#         src = doc.metadata.get("chunk_source", "[]")
#         try:
#             seed_children.extend(json.loads(src))
#         except Exception:
#             # if malformed, skip
#             pass

#     descend(seed_children)

#     return all_results


import json
with open("./essentials/summary_tree.json", "r", encoding="utf-16") as f:
    summary_tr = json.loads(f.read())
# print(summary_tr)
# summary_tr = 

print(raptor_retrieve("What is ARTSENS?",summary_tr))