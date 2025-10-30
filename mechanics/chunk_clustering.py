import os
import numpy as np
import pandas as pd
import umap
import math
import chromadb
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
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
    n_clusters = math.ceil(n_samples // n_clusters)
    best_gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    cluster_labels = best_gmm.fit_predict(reduced_embeddings)

    return np.array(reduced_embeddings), np.array(cluster_labels)


def kmeans_umap_clustering(embeddings, n_neighbors=CONFIG["n_neighbors"], n_components=CONFIG["n_components"], n_clusters=5, random_state: int = 42):
    """UMAP dimensionality reduction followed by KMeans clustering.
    Parameters
    - embeddings: np.ndarray of shape (n_samples, n_dims)
    - n_neighbors, n_components: UMAP params
    - n_clusters: acts as a divisor; actual clusters = ceil(n_samples / n_clusters)
    Returns: (reduced_embeddings, cluster_labels)
    """
    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        return np.array(embeddings), np.zeros(n_samples, dtype=int)

    n_components_safe = max(2, min(n_components, n_samples - 1))
    n_neighbors_safe = max(2, min(n_neighbors, n_samples - 1))

    reducer = umap.UMAP(n_neighbors=n_neighbors_safe, n_components=n_components_safe, init="random", random_state=random_state)
    reduced_embeddings = reducer.fit_transform(embeddings)

    k = max(1, math.ceil(n_samples / max(1, n_clusters)))
    k = min(k, n_samples)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(reduced_embeddings)

    return np.array(reduced_embeddings), np.array(labels)


def spectral_umap_clustering(embeddings, n_neighbors=CONFIG["n_neighbors"], n_components=CONFIG["n_components"], n_clusters=5, assign_labels: str = "kmeans", random_state: int = 42):
    """UMAP dimensionality reduction followed by Spectral Clustering.
    Parameters
    - embeddings: np.ndarray of shape (n_samples, n_dims)
    - n_neighbors, n_components: UMAP params
    - n_clusters: acts as a divisor; actual clusters = ceil(n_samples / n_clusters)
    - assign_labels: 'kmeans' or 'discretize'
    Returns: (reduced_embeddings, cluster_labels)
    """
    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        return np.array(embeddings), np.zeros(n_samples, dtype=int)

    n_components_safe = max(2, min(n_components, n_samples - 1))
    n_neighbors_safe = max(2, min(n_neighbors, n_samples - 1))

    reducer = umap.UMAP(n_neighbors=n_neighbors_safe, n_components=n_components_safe, init="random", random_state=random_state)
    reduced_embeddings = reducer.fit_transform(embeddings)

    k = max(1, math.ceil(n_samples / max(1, n_clusters)))
    k = min(k, n_samples)

    # Use nearest_neighbors affinity to leverage local manifold structure
    spec = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=min(10, max(2, n_samples - 1)),
        assign_labels=assign_labels,
        random_state=random_state,
    )
    labels = spec.fit_predict(reduced_embeddings)

    return np.array(reduced_embeddings), np.array(labels)


