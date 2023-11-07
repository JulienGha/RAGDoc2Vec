from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_documents(model_path, query_vector, topn=5):
    model = Doc2Vec.load(model_path)
    doc_vectors = model.dv.vectors

    similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    related_doc_indices = similarities.argsort()[-topn:][::-1]

    return related_doc_indices


if __name__ == "__main__":
    # This is a placeholder for query vector.
    # In practice, you would generate this from an actual query.
    query_vector = [0.5] * 50  # Example query vector
    indices = retrieve_documents("../models/doc2vec_model.bin", query_vector)

    print(f"Top {len(indices)} related document indices: {indices}")
