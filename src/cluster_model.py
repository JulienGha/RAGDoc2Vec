from transformers import BertTokenizer, BertModel
import torch
import uma
from sklearn.cluster import KMeans
import numpy as np
import pickle


def encode_documents_bert(documents, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    encoded_docs = []
    for doc in documents:
        tokens = tokenizer(doc, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()
        encoded_docs.append(embeddings)

    return np.array(encoded_docs)


def reduce_dimensions_umap(encoded_docs, n_components=5):
    reducer = umap.UMAP(n_components=n_components)
    umap_embeddings = reducer.fit_transform(encoded_docs)
    return umap_embeddings


def cluster_documents_umap(encoded_docs, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(encoded_docs)
    return clusters


def save_models(encoded_docs, umap_model, clusters, save_path):
    with open(save_path + '/encoded_docs.pkl', 'wb') as f:
        pickle.dump(encoded_docs, f)

    with open(save_path + '/umap_model.pkl', 'wb') as f:
        pickle.dump(umap_model, f)

    with open(save_path + '/clusters.pkl', 'wb') as f:
        pickle.dump(clusters, f)


def load_models(load_path):
    with open(load_path + '/encoded_docs.pkl', 'rb') as f:
        encoded_docs = pickle.load(f)

    with open(load_path + '/umap_model.pkl', 'rb') as f:
        umap_model = pickle.load(f)

    with open(load_path + '/clusters.pkl', 'rb') as f:
        clusters = pickle.load(f)

    return encoded_docs, umap_model, clusters


# Training
encoded_docs = encode_documents_bert(documents)
umap_embeddings = reduce_dimensions_umap(encoded_docs)
clusters = cluster_documents_umap(umap_embeddings)

# Saving models
save_models(encoded_docs, umap_embeddings, clusters, save_path='models')

# Loading models
loaded_encoded_docs, loaded_umap_model, loaded_clusters = load_models(load_path='models')
