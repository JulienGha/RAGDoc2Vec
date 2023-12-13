from transformers import BertTokenizer, BertModel
import torch
import umap
from sklearn.cluster import KMeans
import numpy as np
import pickle
from hdbscan import HDBSCAN
import joblib


def encode_documents_bert(documents, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    encoded_docs = []
    for doc in documents:
        tokens = tokenizer(doc, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**tokens)

        # Move the tensor to CPU before converting to NumPy
        embeddings = outputs['last_hidden_state'].mean(dim=1).squeeze().cpu().numpy()
        encoded_docs.append(embeddings)

    return np.array(encoded_docs)


def train_cluster_model(documents, n_clusters=5, algorithm="kmeans"):
    """
    Train a clustering model on the documents.

    Parameters:
    - documents: List of strings, where each string represents the content of a document.
    - n_clusters: Number of clusters to form.
    - algorithm: The clustering algorithm to use ("kmeans" or "hdbscan").

    Returns:
    - encoded_docs: The encoded documents.
    - cluster_model: The trained clustering model.
    - clusters: The cluster assignments for each document.
    """
    if algorithm == "kmeans":
        encoded_docs = reduce_dimensions_umap(encode_documents_bert(documents), n_components=5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(encoded_docs)
        cluster_model = kmeans
    elif algorithm == "hdbscan":
        encoded_docs = encode_documents_bert(documents)
        clusterer = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
        clusters = clusterer.fit_predict(encoded_docs)
        cluster_model = clusterer
    else:
        raise ValueError("Invalid clustering algorithm. Choose either 'kmeans' or 'hdbscan'.")

    # Save UMAP model
    umap_model = train_umap_model(encoded_docs, n_components=5)
    save_umap_model(umap_model, "../models/cluster/umap_model.pkl")

    return encoded_docs, cluster_model, clusters


def train_umap_model(encoded_docs, n_components=5):
    reducer = umap.UMAP(n_components=n_components)
    umap_embeddings = reducer.fit_transform(encoded_docs)
    return reducer


def save_umap_model(umap_model, save_path):
    joblib.dump(umap_model, save_path)


def reduce_dimensions_umap(encoded_docs, n_components=5):
    reducer = umap.UMAP(n_components=n_components)
    umap_embeddings = reducer.fit_transform(encoded_docs)
    return umap_embeddings


def save_cluster_model(encoded_docs, cluster_model, clusters, save_path):
    """
    Save the clustering model and related information.

    Parameters:
    - encoded_docs: The encoded documents.
    - cluster_model: The trained clustering model.
    - clusters: The cluster assignments for each document.
    - save_path: The path to save the model and related information.
    """
    with open(save_path + '/encoded_docs.pkl', 'wb') as f:
        pickle.dump(encoded_docs, f)

    with open(save_path + '/cluster_model.pkl', 'wb') as f:
        pickle.dump(cluster_model, f)

    with open(save_path + '/clusters.pkl', 'wb') as f:
        pickle.dump(clusters, f)


def load_cluster_model(load_path):
    """
    Load the clustering model and related information.

    Parameters:
    - load_path: The path to load the model and related information.

    Returns:
    - encoded_docs: The encoded documents.
    - cluster_model: The loaded clustering model.
    - clusters: The loaded cluster assignments for each document.
    """
    with open(load_path + '/encoded_docs.pkl', 'rb') as f:
        encoded_docs = pickle.load(f)

    with open(load_path + '/cluster_model.pkl', 'rb') as f:
        cluster_model = pickle.load(f)

    with open(load_path + '/clusters.pkl', 'rb') as f:
        clusters = pickle.load(f)

    return encoded_docs, cluster_model, clusters
