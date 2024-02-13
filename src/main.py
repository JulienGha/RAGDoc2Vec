import json
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from retriever import retrieve_documents_doc2vec, retrieve_documents_bert, retrieve_tfidf
from doc2vec import train_doc2vec
from bert import train_bert_model
from tfidf import train_tfidf
from language_model import prompt_opti, generate_response
import os


def main(pdf_directory, context_size=60):
    train_new_model = ""
    if not os.path.exists('../data/raw'):
        os.makedirs('../data/raw')
    if not os.path.exists('../data/pdf'):
        os.makedirs('../data/pdf')
    if not os.path.exists('../models/bert'):
        os.makedirs('../models/bert')

    # Process documents in the specified directory
    if pdf_directory:
        train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()

        if train_new_model == 'yes':
            list_doc = []
            list_files = []

            print("Processing files...")
            for filename in os.listdir(pdf_directory):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(pdf_directory, filename)
                    list_files.append(file_path)
                    convert_pdf_into_json(filename, context_size)
                    filename = filename.replace(".pdf", "")
                    processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + filename + '.json'),
                                                                 filename)
                    list_doc.extend(processed_docs)

            print("Files processed...")

            # Convert the preprocessed data into a format compatible with the retrieve function
            # Join the words in the TaggedDocument to form the full text of the document
            model_choice = ""

            while model_choice not in ["doc", "bert", "tfidf"]:
                model_choice = input("Do you want doc2vec, BERT or tfidf? (doc/bert/tfidf): ").strip().lower()

                if model_choice == "doc":
                    print("Training model...")

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/doc2vec', exist_ok=True)

                    # Train the Doc2Vec model
                    train_doc2vec(list_doc)

                    # Save docs for future use
                    with open('../models/doc2vec/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    print("Model trained")

                elif model_choice == "bert":
                    print("Training model...")

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/bert', exist_ok=True)

                    # Train the BERT model and get encoded documents
                    documents = [" ".join(doc.words) for doc in list_doc]
                    train_bert_model(documents)

                    # Save docs for future use
                    with open('../models/bert/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    print("Model trained")

                elif model_choice == "tfidf":
                    print("Training model...")

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/tfidf', exist_ok=True)

                    train_tfidf(list_doc)

                    # Save the model for future use
                    with open('../models/tfidf/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    print("Model trained")

        elif train_new_model == "no":
            model_choice = ""

            while model_choice not in ["doc", "bert", "tfidf"]:
                model_choice = input(
                    "Do you want to load doc2vec, BERT or tfidf? (doc/bert/tfidf): ").strip().lower()

                if model_choice == "doc":
                    print("Loading model...")
                    if os.path.exists('../models/doc2vec/last_file.json'):
                        # Load an existing model
                        with open('../models/doc2vec/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break
                    print("Model loaded")

                elif model_choice == "bert":
                    print("Loading model...")
                    if os.path.exists('../models/bert/last_file.json'):
                        # Load an existing model
                        with open('../models/bert/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break
                    print("Model loaded")

                elif model_choice == "tfidf":
                    if os.path.exists('../models/tfidf/last_file.json'):
                        # Load an existing model
                        with open('../models/tfidf/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break
                    print("Model loaded")

    else:
        model_choice = ""

        while model_choice not in ["doc", "bert", "tfidf"]:
            model_choice = input("No file in input, going directly into loading. "
                                 "Do you want to load doc2vec, BERT or tfidf? (doc/bert/tfidf): ").strip().lower()

            if model_choice == "doc":
                print("Loading model...")
                if os.path.exists('../models/doc2vec/last_file.json'):
                    # Load an existing model
                    with open('../models/doc2vec/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
                    break
                print("Model loaded")

            elif model_choice == "bert":
                print("Loading model...")
                if os.path.exists('../models/bert/last_file.json'):
                    # Load an existing model
                    with open('../models/bert/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
                    break
                print("Model loaded")

            elif model_choice == "tfidf":
                print("Loading Model...")
                if os.path.exists('../models/tfidf/last_file.json'):
                    # Load an existing model
                    with open('../models/tfidf/last_file.json', 'r') as file:
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
            retrieved_docs_nopo = retrieve_documents_bert(query, documents)
            retrieved_docs = retrieve_documents_bert(optimized_query, documents)

        elif model_choice == "tfidf":
            retrieved_docs_nopo = retrieve_tfidf(query, documents)
            retrieved_docs = retrieve_tfidf(optimized_query, documents)

        # Concatenate documents content to form the context for generation
        context_nopo = " ".join([content for _, content in retrieved_docs_nopo])
        context_opti = " ".join([content for _, content in retrieved_docs])

        print("Answer without RAG & no po: ")
        generate_response("", query)

        print("Answer with RAG & no po: ")
        generate_response(context_nopo, query)

        print("With RAG and optimization: ")
        generate_response(context_opti, query)


if __name__ == "__main__":
    main("../data/pdf")
