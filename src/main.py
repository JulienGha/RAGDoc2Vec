import json
from gensim.models import Doc2Vec
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from doc2vec import train_doc2vec, retrieve_documents_doc2vec
from faiss import create_vector_db
from generate import generate_response  # Import the generate_response function
from bert import retrieve_documents_bert, train_bert_model, save_bert_model, load_bert_model
import os


def main(files):
    train_new_model = ""
    # Process documents if files are provided
    if files:
        train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()
        if train_new_model == 'yes':
            list_doc = []
            list_files = []
            for file in files:
                list_files.append('../data/raw/' + file + '.pdf')
                convert_pdf_into_json(file)
                processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + file + '.json'), file)
                list_doc.extend(processed_docs)
            # Convert the preprocessed data into a format compatible with the retrieve function
            # Join the words in the TaggedDocument to form the full text of the document
            model_choice = ""
            while model_choice not in ["doc", "bert", "faiss"]:
                model_choice = input("Do you want doc2vec, BERT or faiss? (doc/bert/faiss): ").strip().lower()
                if model_choice == "doc":
                    model_path = "../models/doc2vec/doc2vec_model.bin"
                    with open('../models/doc2vec/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)
                    # Train the Doc2Vec model
                    model = train_doc2vec(list_doc)
                    # Save the model for future use
                    model.save(model_path)
                elif model_choice == "bert":
                    # Train the BERT model and get encoded documents
                    documents = [" ".join(doc.words) for doc in list_doc]
                    encoded_docs = train_bert_model(documents)
                    # Save the model for future use
                    with open('../models/bert/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)
                    save_bert_model(encoded_docs)
                elif model_choice == "faiss":
                    create_vector_db(list_files)
                    model = "../models/faiss_db"
        elif train_new_model == "no":
            model_choice = ""
            while model_choice not in ["doc", "bert", "faiss"]:
                model_choice = input("Do you want to load doc2vec, BERT or faiss? (doc/bert/faiss): ").strip().lower()
                if model_choice == "doc":
                    model_path = "../models/doc2vec/doc2vec_model.bin"
                    if os.path.exists(model_path):
                        # Load an existing model
                        model = Doc2Vec.load(model_path)
                        with open('../models/doc2vec/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("this model doesnt exist, plz train a new one")
                        break
                elif model_choice == "bert":
                    model_path = "../models/bert/bert_model.pkl"
                    if os.path.exists(model_path):
                        # Load an existing model
                        encoded_docs = load_bert_model()
                        with open('../models/bert/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("this model doesnt exist, plz train a new one")
                        break
                elif model_choice == "faiss":
                    model_path = "../models/faiss_db"
                    if os.path.exists(model_path):
                        # Load an existing model
                        model = Doc2Vec.load(model_path)
                    else:
                        print("this model doesnt exist, plz train a new one")
                        break
    else:
        model_choice = ""
        while model_choice not in ["doc", "bert", "faiss"]:
            model_choice = input("Do you want to load doc2vec, BERT or faiss? (doc/bert/faiss): ").strip().lower()
            if model_choice == "doc":
                model_path = "../models/doc2vec/doc2vec_model.bin"
                if os.path.exists(model_path):
                    # Load an existing model
                    model = Doc2Vec.load(model_path)
                    with open('../models/doc2vec/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("this model doesnt exist, plz train a new one")
                    break
            elif model_choice == "bert":
                model_path = "../models/bert/bert_model.pkl"
                if os.path.exists(model_path):
                    # Load an existing model
                    # Load the encoded documents
                    encoded_docs = load_bert_model()
                    with open('../models/bert/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("this model doesnt exist, plz train a new one")
                    break
            elif model_choice == "faiss":
                model_path = "../models/faiss_db"
                if os.path.exists(model_path):
                    # Load an existing model
                    model = Doc2Vec.load(model_path)
                else:
                    print("this model doesnt exist, plz train a new one")
                    break
    if train_new_model == "yes":
        documents = [" ".join(doc.words) for doc in list_doc]
    else:
        documents = [" ".join(doc["words"]) for doc in list_doc]
    # User input for retrieval
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        if model_choice == "doc":
            # Preprocess and infer the query vector
            query_vector = model.infer_vector(query.split())
            # Retrieve documents and their contents using the trained model
            retrieved_docs = retrieve_documents_doc2vec(model_choice, query_vector, documents)
        elif model_choice == "bert":
            retrieved_docs = retrieve_documents_bert(query, encoded_docs, documents)
        elif model_choice == "faiss":
            return

        # Concatenate documents content to form the context for generation
        context = " ".join([content for _, content in retrieved_docs])

        # Generate a response using the context
        response = generate_response(context)
        print(f"Generated response: {response}")


if __name__ == "__main__":
    main(["cognitive_neuropsycho_schizo"])
