import os
import numpy as np
import pandas as pd
import umap
import math
import chromadb
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "n_neighbors": 15,
    "n_components": 64,
    "min_dist": 0.1,
    "metric": "cosine"
}

def gmm_umap_clustering(embeddings, n_neighbors=CONFIG["n_neighbors"], n_components=CONFIG["n_components"], n_clusters=5):
    n_samples = embeddings.shape[0]
    n_components_safe = min(n_components, n_samples - 1)
    n_neighbors_safe = min(n_neighbors, n_samples - 1)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors_safe, n_components=n_components_safe, init="random", random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    # print(reduced_embeddings.shape, embeddings.shape)
    n_clusters = math.ceil(n_samples // n_clusters)
    # print(n_clusters)
    best_gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    cluster_labels = best_gmm.fit_predict(reduced_embeddings)

    return np.array(reduced_embeddings), np.array(cluster_labels)


# vectorstore = Chroma(persist_directory="./chroma_store")
# collection = vectorstore._collection
# results = collection.get(include=["embeddings", "documents", "metadatas"])
# embeddings = np.array(results["embeddings"])
# docs = results["documents"]
# metadatas = results["metadatas"]

# embeddings = StandardScaler().fit_transform(embeddings)
# umap_embeddings, clusters = gmm_umap_clustering(embeddings, n_neighbors=15, n_components=64)
# print(umap_embeddings.shape)


# n_clusters = math.ceil(len(embeddings) / 5)
# best_gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)

# cluster_labels = best_gmm.fit_predict(umap_embeddings)

# cluster_info = {
#     "file_id": [meta["id"] for meta in metadatas],
#     "cluster": cluster_labels
# }