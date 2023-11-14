import json
from gensim.models import Doc2Vec
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from retrieve import retrieve_documents
from doc2vec import train_doc2vec
from generate import generate_response  # Import the generate_response function
import os


def main(files):

    model_path = "../models/doc2vec_model.bin"
    # Process documents if files are provided
    if files:
        list_doc = []
        for file in files:
            convert_pdf_into_json(file)
            processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + file + '.json'), custom_tags=True)
            with open('../data/processed/' + file + '.json', "w") as file_p:
                json.dump([{"words": doc.words, "tags": doc.tags} for doc in processed_docs], file_p)
            list_doc.extend(processed_docs)

    # Ask the user if they want to train a new model
    train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()
    if train_new_model == 'yes':
        model_choice = ""
        if files:
            list_doc = []
            for file in files:
                convert_pdf_into_json(file)
                processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + file + '.json'))
                with open('../data/processed/' + file + '.json', "w") as file_p:
                    json.dump([doc.words for doc in processed_docs], file_p)
                for string in processed_docs:
                    list_doc.append(string)
            while model_choice != "doc" or model_choice != "bert":
                model_choice = input("Do you want doc2vec or BERT? (doc/bert): ").strip().lower()
                if model_choice == "doc":
                    # Train the Doc2Vec model
                    model = train_doc2vec(list_doc)
                    # Save the model for future use
                    model.save(model_path)
                elif model_choice == "bert":
                    # Train the Doc2Vec model
                    model = train_doc2vec(list_doc)
                    # Save the model for future use
                    model_path = "../models/bert_model.bin"
                    model.save(model_path)
        else:
            print("can train a model with no files detected or in input")
    elif train_new_model == "no" and os.path.exists(model_path):
        # Load an existing model
        model = Doc2Vec.load(model_path)
    elif train_new_model == "no":
        print(f"No existing model found at {model_path}. Training a new model.")
        model = train_doc2vec(list_doc)
        model.save(model_path)

    # Convert the preprocessed data into a format compatible with the retrieve function
    # Join the words in the TaggedDocument to form the full text of the document
    documents = [" ".join(doc.words) for doc in list_doc]

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
    main(["cognitive_neuropsycho_schizo"])
