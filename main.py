from sklearn.metrics.pairwise import cosine_similarity
import doc2vec
def retrieve_documents(model, query, documents, topn=5):
    query_vector = model.infer_vector(word_tokenize(query.lower()))
    doc_vectors = [model.dv[str(i)] for i in range(len(documents))]
    similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    related_docs_indices = similarities.argsort()[-topn:][::-1]
    return [(documents[i], similarities[i]) for i in related_docs_indices]

# Retrieve for a sample query
retrieved_docs = retrieve_documents(doc2vec_model, queries[0], documents)


