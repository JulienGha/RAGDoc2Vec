import json
from gensim.models import Doc2Vec
from preprocess import preprocess_data, load_data
from retrieve import retrieve_documents
from doc2vec import train_doc2vec
from generate import generate_response  # Import the generate_response function
import os


def main(file):
    model_path = "../models/doc2vec_model.bin"
    data_path = '../data/processed/' + file + '.json'  # Adjust the path to where your data is

    # Load and preprocess the data
    preprocessed_data = preprocess_data(load_data(data_path))

    # Ask the user if they want to train a new model
    train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()
    if train_new_model == 'yes':
        # Train the Doc2Vec model
        model = train_doc2vec(preprocessed_data)

        # Save the model for future use
        model.save(model_path)
    elif os.path.exists(model_path):
        # Load an existing model
        model = Doc2Vec.load(model_path)
    else:
        print(f"No existing model found at {model_path}. Training a new model.")
        model = train_doc2vec(preprocessed_data)
        model.save(model_path)

    # Convert the preprocessed data into a format compatible with the retrieve function
    # Join the words in the TaggedDocument to form the full text of the document
    documents = [" ".join(doc.words) for doc in preprocessed_data]

    # User input for retrieval
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        # Preprocess and infer the query vector
        query_vector = model.infer_vector(query.split())

        # Retrieve documents and their contents using the trained model
        retrieved_docs = retrieve_documents("../models/doc2vec_model.bin", query_vector, documents)

        # Concatenate documents content to form the context for generation
        context = " ".join([content for _, content in retrieved_docs])

        # Generate a response using the context
        response = generate_response(context)
        print(f"Generated response: {response}")


if __name__ == "__main__":
    main()
