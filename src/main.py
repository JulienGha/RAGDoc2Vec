import json
from gensim.models import Doc2Vec
from preprocess import preprocess_data, load_data
from retrieve import retrieve_documents
from doc2vec import train_doc2vec


def main():
    # Load and preprocess the data
    data_path = '../data/quran_en.json'  # Adjust the path to where your data is
    preprocessed_data = preprocess_data(load_data(data_path))

    # Train the Doc2Vec model
    model = train_doc2vec(preprocessed_data)

    # Save the model for future use
    model.save("../models/doc2vec_model.bin")

    # User input for retrieval
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        # Retrieve documents using the trained model
        retrieved_docs = retrieve_documents(model, query, preprocessed_data)

        # Display results
        for doc_id, similarity in retrieved_docs:
            print(f"Doc ID: {doc_id}, Similarity: {similarity}")
            # Fetch the actual document using doc_id if necessary


if __name__ == "__main__":
    main()
