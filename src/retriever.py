import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Doc2Vec


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_documents_bert(query, encoded_docs, documents, topn=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

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
    print(f"The found document are: {related_documents}")
    return related_documents


# Assume `documents` is a list of strings representing the content of each document
def retrieve_documents_doc2vec(query_vector, documents, topn=5):
    model = Doc2Vec.load("../models/doc2vec/doc2vec_model.bin")
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
    print(f"The found document are: {related_documents}")
    return related_documents
