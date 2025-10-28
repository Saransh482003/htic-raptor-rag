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
vectorstore = Chroma(
    persist_directory="./chroma_store",
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

    # Retrieve top-k root-level summaries
    root_results = vectorstore.similarity_search(
        query,
        k=top_k_root,
        filter={"id": {"$in": root_ids}}
    )

    # Store all retrieved documents from all levels
    all_results = []
    all_results.extend(root_results)  # Add root-level summaries

    def descend(children):
        if all(cid.endswith("level_0") for cid in children):
            child_results = vectorstore.similarity_search(
                query,
                k=len(children),
                filter={"id": {"$in": children}}
            )
            all_results.extend(child_results)  # Add leaf chunks
            return
        
        child_results = vectorstore.similarity_search(
            query,
            k=min(top_k_children, len(children)),
            filter={"id": {"$in": children}}
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
    
    return all_results


query = "How does the SphygmoCor XCEL measure blood pressure?"
results = raptor_retrieve(query,summary_tree)
