import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize
import umap
import pickle
import joblib
import numpy as np

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to save the BERT model's encoded documents and UMAP model
def save_bert_model(encoded_docs, umap_model, path="../models/bert/bert_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"encoded_docs": encoded_docs, "umap_model": umap_model}, f)


# Function to train a BERT model (for encoding documents)
def train_bert_model(documents, use_umap=True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Encoding the documents
    encoded_docs = []
    for doc in documents:
        encoded_input = tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        encoded_docs.append(model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())

    # Convert the list of NumPy arrays to a single NumPy array before creating a PyTorch tensor
    encoded_docs = np.array(encoded_docs)

    # Apply UMAP to the encoded document vectors if specified
    if use_umap:
        reducer = umap.UMAP(n_neighbors=45, n_components=2, min_dist=0.1, metric='cosine')
        encoded_docs = reducer.fit_transform(encoded_docs)
        encoded_docs = normalize(encoded_docs, axis=0, norm='l2')
        # Save the UMAP model
        joblib.dump(reducer, '../models/bert/umap_model.sav')

    save_bert_model(encoded_docs, reducer)
    return encoded_docs
