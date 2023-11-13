from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


# Assume `documents` is a list of strings representing the content of each document
def retrieve_documents(model_path, query_vector, documents, topn=5):
    model = Doc2Vec.load(model_path)
    doc_vectors = model.dv.vectors

    similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    related_doc_indices = similarities.argsort()[-topn:][::-1]

    # Fetch the actual documents using the indices
    related_documents = [(idx, documents[idx]) for idx in related_doc_indices]
    return related_documents


if __name__ == "__main__":
    # Placeholder for documents and query vector
    documents = ["document 1 content", "document 2 content", ...]  # Replace with actual document contents
    query_vector = [0.5] * 50  # Example query vector

    related_documents = retrieve_documents("../models/doc2vec_model.bin", query_vector, documents)
    for idx, content in related_documents:
        print(f"Doc ID: {idx}, Content: {content}")
