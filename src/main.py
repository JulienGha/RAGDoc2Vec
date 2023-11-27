import json
from gensim.models import Doc2Vec
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from retriever import retrieve_documents_doc2vec, retrieve_documents_bert
from doc2vec import train_doc2vec
from bert import train_bert_model, save_bert_model, load_bert_model
from faiss import create_vector_db
from language_model import prompt_opti, generate_response
import os


def main(files):
    train_new_model = ""
    # Process documents if files are provided
    if files:
        train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()
        if train_new_model == 'yes':
            list_doc = []
            list_files = []
            print("Processing files...")
            for file in files:
                list_files.append('../data/pdf/' + file + '.pdf')
                convert_pdf_into_json(file)
                processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + file + '.json'), file)
                list_doc.extend(processed_docs)
            print("Files processed...")
            # Convert the preprocessed data into a format compatible with the retrieve function
            # Join the words in the TaggedDocument to form the full text of the document
            model_choice = ""
            while model_choice not in ["doc", "bert", "faiss"]:
                model_choice = input("Do you want doc2vec, BERT or faiss? (doc/bert/faiss): ").strip().lower()
                if model_choice == "doc":
                    print("Training model...")
                    model_path = "../models/doc2vec/doc2vec_model.bin"
                    with open('../models/doc2vec/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)
                    # Train the Doc2Vec model
                    model = train_doc2vec(list_doc)
                    # Save the model for future use
                    model.save(model_path)
                    print("Model trained")
                elif model_choice == "bert":
                    print("Training model...")
                    # Train the BERT model and get encoded documents
                    documents = [" ".join(doc.words) for doc in list_doc]
                    encoded_docs = train_bert_model(documents)
                    # Save the model for future use
                    with open('../models/bert/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)
                    save_bert_model(encoded_docs)
                    print("Model trained")
                elif model_choice == "faiss":
                    print("Training model...")
                    create_vector_db()
                    model = "../models/faiss_db"
                    print("Model trained")
        elif train_new_model == "no":
            model_choice = ""
            while model_choice not in ["doc", "bert", "faiss"]:
                model_choice = input("Do you want to load doc2vec, BERT or faiss? (doc/bert/faiss): ").strip().lower()
                if model_choice == "doc":
                    print("Loading model...")
                    model_path = "../models/doc2vec/doc2vec_model.bin"
                    if os.path.exists(model_path):
                        # Load an existing model
                        model = Doc2Vec.load(model_path)
                        with open('../models/doc2vec/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("this model doesnt exist, plz train a new one")
                        break
                    print("Model loaded")
                elif model_choice == "bert":
                    print("Loading model...")
                    model_path = "../models/bert/bert_model.pkl"
                    if os.path.exists(model_path):
                        # Load an existing model
                        encoded_docs = load_bert_model()
                        with open('../models/bert/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("this model doesnt exist, plz train a new one")
                        break
                    print("Model loaded")
                elif model_choice == "faiss":
                    print("Loading model...")
                    model_path = "../models/faiss_db"
                    if os.path.exists(model_path):
                        # Load an existing model
                        model = Doc2Vec.load(model_path)
                    else:
                        print("this model doesnt exist, plz train a new one")
                        break
                    print("Model loaded")
    else:
        model_choice = ""
        while model_choice not in ["doc", "bert", "faiss"]:
            model_choice = input("No file in input, going directly into loading."
                                 "Do you want to load doc2vec, BERT or faiss? (doc/bert/faiss): ").strip().lower()
            if model_choice == "doc":
                print("Loading model...")
                model_path = "../models/doc2vec/doc2vec_model.bin"
                if os.path.exists(model_path):
                    # Load an existing model
                    model = Doc2Vec.load(model_path)
                    with open('../models/doc2vec/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("this model doesnt exist, plz train a new one")
                    break
                print("Model loaded")
            elif model_choice == "bert":
                print("Loading model...")
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
                print("Model loaded")
            elif model_choice == "faiss":
                print("Loading model...")
                model_path = "../models/faiss_db"
                if os.path.exists(model_path):
                    # Load an existing model
                    model = Doc2Vec.load(model_path)
                else:
                    print("this model doesnt exist, plz train a new one")
                    break
                print("Model loaded")
    if train_new_model == "yes":
        documents = [" ".join(doc.words) for doc in list_doc]
    else:
        documents = [" ".join(doc["words"]) for doc in list_doc]
    # User input for retrieval
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        optimized_query = prompt_opti(query)
        if model_choice == "doc":
            query_words1 = optimized_query[1].split()
            query_vector1 = model.infer_vector(query_words1)
            retrieved_docs1 = retrieve_documents_doc2vec(query_vector1, documents)
            query_words2 = optimized_query[0].split()
            query_vector2 = model.infer_vector(query_words2)
            retrieved_docs2 = retrieve_documents_doc2vec(query_vector2, documents)
        elif model_choice == "bert":
            retrieved_docs1 = retrieve_documents_bert(optimized_query[1], encoded_docs, documents)
            retrieved_docs2 = retrieve_documents_bert(optimized_query[0], encoded_docs, documents)
        elif model_choice == "faiss":
            return

        # Concatenate documents content to form the context for generation
        context1 = " ".join([content for _, content in retrieved_docs1])
        print(context1)
        context2 = " ".join([content for _, content in retrieved_docs2])
        print(context2)

        print("Answer without RAG: ")
        response = generate_response("", query)
        print(f"Generated response: {response}")

        print("Answer with one query optimization: ")
        response = generate_response(context1, query)
        print(f"Generated response: {response}")

        print("Answer with 2 query optimization: ")
        response = generate_response(context2, query)
        print(f"Generated response: {response}")


if __name__ == "__main__":
    main(["cognitive_neuropsycho_schizo"])
