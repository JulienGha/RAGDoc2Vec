import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Doc2Vec
from bert import load_bert_model
import json

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_documents_bert(query, encoded_docs, documents, topn=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    print(query)

    # Encoding the query
    encoded_query = tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_vector = query_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # Move encoded_docs to GPU
    encoded_docs = torch.tensor(encoded_docs).to(device)

    # Calculating similarities
    similarities = cosine_similarity([query_vector], encoded_docs.cpu()).flatten()
    related_doc_indices = similarities.argsort()[-topn:][::-1]

    surrounding_docs_idx = []
    # Fetch the actual documents using the indices
    for idx in related_doc_indices:
        if idx > 0:
            surrounding_docs_idx.append(idx - 1)
        surrounding_docs_idx.append(idx)
        if idx < len(documents) - 1:
            surrounding_docs_idx.append(idx + 1)
    # Fetch the actual documents using the indices
    related_documents = [(idx, documents[idx]) for idx in surrounding_docs_idx]
    print(f"Found documents: {related_documents}")
    return related_documents


# Assume `documents` is a list of strings representing the content of each document
def retrieve_documents_doc2vec(query, documents, topn=5):
    model = Doc2Vec.load("../models/doc2vec/doc2vec_model.bin")
    print(query)
    query_vector = model.infer_vector(query.split())
    doc_vectors = model.dv.vectors

    # Move query_vector to GPU
    query_vector = torch.tensor(query_vector).to(device).cpu().numpy()

    similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    related_doc_indices = similarities.argsort()[-topn:][::-1]

    surrounding_docs_idx = []
    # Fetch the actual documents using the indices
    for idx in related_doc_indices:
        if idx > 0:
            surrounding_docs_idx.append(idx - 1)
        surrounding_docs_idx.append(idx)
        if idx < len(documents) - 1:
            surrounding_docs_idx.append(idx + 1)
    # Fetch the actual documents using the indices
    related_documents = [(idx, documents[idx]) for idx in surrounding_docs_idx]
    print(f"Found documents: {related_documents}")
    return related_documents


def retrieve_documents_cluster(query_vector, umap_model, kmeans_model, encoded_docs, documents, topn=5):
    # Find the nearest cluster to the query vector using UMAP
    nearest_cluster = umap_model.transform([query_vector])

    # Assign the query to the nearest cluster using k-means
    cluster_assignment = kmeans_model.predict(nearest_cluster)

    # Get indices of documents in the assigned cluster
    cluster_indices = np.where(kmeans_model.labels_ == cluster_assignment)[0]

    # Calculate cosine similarity between the query vector and documents in the cluster
    similarities = cosine_similarity([query_vector], encoded_docs[cluster_indices]).flatten()

    # Retrieve topn documents based on similarity
    topn_indices = similarities.argsort()[-topn:][::-1]
    topn_indices = cluster_indices[topn_indices]

    # Fetch the actual documents using the indices
    topn_documents = [(idx, documents[idx]) for idx in topn_indices]
    print(f"Found top {topn} documents in the assigned cluster: {topn_documents}")
    return topn_documents



"""with open('../models/bert/last_file.json', 'r') as file:
    list_doc = json.load(file)


list_doc = retrieve_documents_bert("Hallucination refers to the perception of sensory stimuli, such as sights, sounds, "
                                   "or sensations, that are not present in reality and originate from the mind",
                                    load_bert_model(), list_doc, topn=5)

for element in list_doc:
    print(element)"""

