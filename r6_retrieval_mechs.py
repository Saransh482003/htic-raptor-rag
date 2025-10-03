import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore")

# Load vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embeddings
)

# Load original summary tree (for hierarchy reference if needed)
with open("./essentials/summary_tree.json", "r", encoding="utf-16") as f:
    summary_tree = json.load(f)


def raptor_retrieve(query, top_k_root=1, top_k_children=2):
    """
    Perform RAPTOR-style hierarchical retrieval.
    """

    root_level_map = {}
    for file_id, file_data in summary_tree.items():
        levels = list(file_data["levels"].keys())
        max_level = max(levels, key=lambda x: int(x.split("_")[1]))
        for chunk in file_data["levels"][max_level]:
            root_level_map[chunk["id"]] = int(max_level.split("_")[1])

    root_ids = list(root_level_map.keys())

    root_results = vectorstore.similarity_search(
        query,
        k=top_k_root,
        filter={"id": {"$in": root_ids}}
    )

    def descend(children):
        if all(cid.endswith("level_0") for cid in children):
            child_results = vectorstore.similarity_search(
                query,
                k=len(children),
                filter={"id": {"$in": children}}
            )
            return child_results
        
        child_results = vectorstore.similarity_search(
            query,
            k=min(top_k_children, len(children)),
            filter={"id": {"$in": children}}
        )
        next_children = []
        for doc in child_results:
            next_children.extend(json.loads(doc.metadata["chunk_source"]))
        return descend(next_children)

    children = []
    for doc in root_results:
        children.extend(json.loads(doc.metadata["chunk_source"]))
    
    return descend(children)


query = "How does the SphygmoCor XCEL measure blood pressure?"
results = raptor_retrieve(query)
