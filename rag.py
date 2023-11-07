from doc2vec_train import train_doc2vec
from preprocess import preprocess_data, load_data
from retrieve import retrieve_documents
from generate import generate_response


def rag_process(query, doc2vec_model, documents):
    # Retrieve documents for the query
    query_vector = doc2vec_model.infer_vector(query.split())
    doc_indices = retrieve_documents("../models/doc2vec_model.bin", query_vector)

    # Prepare context for generation
    context = ' '.join([documents[i] for i in doc_indices])
    context_with_query = f"Query: {query}\nContext: {context}\nAnswer:"

    # Generate response
    response = generate_response(context_with_query)
    return response


if __name__ == "__main__":
    # Load and preprocess documents
    documents = load_data('../data/documents.json')
    processed_docs = preprocess_data(documents)

    # Train Doc2Vec model (or load if already trained)
    doc2vec_model = train_doc2vec(processed_docs)

    # Get the query from user input
    query = input("Please enter your query: ")

    response = rag_process(query, doc2vec_model, documents)
    print(f"RAG Response: {response}")
