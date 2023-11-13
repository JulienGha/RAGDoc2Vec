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


