from transformers import DPRReader
import pickle
import os


def train_dpr_model(documents, model_name='facebook/dpr-reader-single-nq-base'):
    # Load the pre-trained DPR model
    model = DPRReader.from_pretrained(model_name)

    # Encode each document
    document_embeddings = [model.encode_passage(doc) for doc in documents]

    # You may perform additional training steps here if needed

    return document_embeddings, model


def save_dpr_model(document_embeddings, dpr_model, save_path):
    """
    Save the DPR model and related information.

    Parameters:
    - document_embeddings: The encoded documents.
    - dpr_model: The trained DPR model.
    - save_path: The path to save the model and related information.
    """
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'document_embeddings.pkl'), 'wb') as f:
        pickle.dump(document_embeddings, f)

    dpr_model.save_pretrained(save_path)


def load_dpr_model(load_path):
    """
    Load the DPR model and related information.

    Parameters:
    - load_path: The path to load the model and related information.

    Returns:
    - document_embeddings: The encoded documents.
    - dpr_model: The loaded DPR model.
    """
    with open(os.path.join(load_path, 'document_embeddings.pkl'), 'rb') as f:
        document_embeddings = pickle.load(f)

    dpr_model = DPRReader.from_pretrained(load_path)

    return document_embeddings, dpr_model
