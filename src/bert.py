from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# Function to train a BERT model (for encoding documents)
def train_bert_model(documents):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encoding the documents
    encoded_docs = []
    for doc in documents:
        encoded_input = tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        encoded_docs.append(model_output.last_hidden_state.mean(dim=1).squeeze().numpy())

    return encoded_docs


# Function to save the BERT model's encoded documents
def save_bert_model(encoded_docs, path="../models/bert/bert_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(encoded_docs, f)


# Function to load the BERT model's encoded documents
def load_bert_model(path="../models/bert/bert_model.pkl"):
    with open(path, "rb") as f:
        encoded_docs = pickle.load(f)
    return encoded_docs


# Function to retrieve documents using BERT encoded vectors
def retrieve_documents_bert(query, encoded_docs, documents, topn=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encoding the query
    encoded_query = tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_vector = query_output.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Calculating similarities
    similarities = cosine_similarity([query_vector], encoded_docs).flatten()
    related_doc_indices = similarities.argsort()[-topn:][::-1]

    # Fetch the actual documents using the indices
    related_documents = [(idx, documents[idx]) for idx in related_doc_indices]
    print(related_documents)
    return related_documents
