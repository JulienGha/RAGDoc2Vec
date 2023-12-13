import json
from gensim.models import Doc2Vec
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from retriever import retrieve_documents_doc2vec, retrieve_documents_bert, retrieve_documents_cluster
from doc2vec import train_doc2vec
from bert import train_bert_model, save_bert_model, load_bert_model
from language_model import prompt_opti, generate_response
import os
from cluster_model import load_cluster_model, save_cluster_model, train_cluster_model


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

            while model_choice not in ["doc", "bert", "cluster"]:
                model_choice = input("Do you want doc2vec, BERT or cluster? (doc/bert/cluster): ").strip().lower()

                if model_choice == "doc":
                    print("Training model...")
                    model_path = "../models/doc2vec/doc2vec_model.bin"

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/doc2vec', exist_ok=True)

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

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/bert', exist_ok=True)

                    # Save the model for future use
                    with open('../models/bert/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    save_bert_model(encoded_docs)
                    print("Model trained")

                elif model_choice == "cluster":

                    print("Training model...")

                    # Train the cluster model and get encoded documents
                    documents = [" ".join(doc.words) for doc in list_doc]
                    encoded_docs, cluster_model, clusters = train_cluster_model(documents)

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/cluster', exist_ok=True)

                    # Save the model for future use

                    with open('../models/cluster/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    # Correct function call with all required arguments
                    save_cluster_model(encoded_docs, cluster_model, clusters, "../models/cluster")

                    print("Model trained")

        elif train_new_model == "no":
            model_choice = ""

            while model_choice not in ["doc", "bert", "cluster"]:
                model_choice = input(
                    "Do you want to load doc2vec, BERT or cluster? (doc/bert/cluster): ").strip().lower()

                if model_choice == "doc":
                    print("Loading model...")
                    model_path = "../models/doc2vec/doc2vec_model.bin"

                    if os.path.exists(model_path):
                        # Load an existing model
                        with open('../models/doc2vec/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
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
                        print("This model doesn't exist, please train a new one.")
                        break

                    print("Model loaded")

                elif model_choice == "cluster":
                    print("Loading model...")
                    model_path = "../models/cluster/"

                    if os.path.exists(model_path):
                        # Load an existing model
                        encoded_docs, cluster_model, clusters = load_cluster_model(model_path)
                        with open('../models/cluster/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break

                    print("Model loaded")

    else:
        model_choice = ""

        while model_choice not in ["doc", "bert", "cluster"]:
            model_choice = input("No file in input, going directly into loading."
                                 "Do you want to load doc2vec, BERT or cluster? (doc/bert/cluster): ").strip().lower()

            if model_choice == "doc":
                print("Loading model...")
                model_path = "../models/doc2vec/doc2vec_model.bin"

                if os.path.exists(model_path):
                    # Load an existing model
                    model = Doc2Vec.load(model_path)
                    with open('../models/doc2vec/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
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
                    print("This model doesn't exist, please train a new one.")
                    break

                print("Model loaded")

            elif model_choice == "cluster":
                print("Loading model...")
                model_path = "../models/cluster/cluster_model.pkl"

                if os.path.exists(model_path):
                    # Load an existing model
                    encoded_docs, cluster_model, clusters = load_cluster_model(model_path)
                    with open('../models/cluster/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
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
            retrieved_docs_nopo = retrieve_documents_doc2vec(query, documents)
            retrieved_docs = retrieve_documents_doc2vec(optimized_query, documents)

        elif model_choice == "bert":
            retrieved_docs_nopo = retrieve_documents_bert(query, encoded_docs, documents)
            retrieved_docs = retrieve_documents_bert(optimized_query, encoded_docs, documents)

        elif model_choice == "cluster":
            retrieved_docs_nopo = retrieve_documents_cluster(query, umap_model, kmeans_model, encoded_docs,
                                                             documents)
            retrieved_docs = retrieve_documents_cluster(optimized_query, umap_model, kmeans_model, encoded_docs,
                                                        documents)

        # Concatenate documents content to form the context for generation
        context_nopo = " ".join([content for _, content in retrieved_docs_nopo])
        context_opti = " ".join([content for _, content in retrieved_docs])

        """print("Answer without RAG & no po: ")
        generate_response("", query)

        print("Answer with RAG & no po: ")
        generate_response(context_nopo, query)"""

        print("With RAG and optimization: ")
        generate_response(context_opti, query)


if __name__ == "__main__":
    main(["cognitive_neuropsycho_schizo"])
