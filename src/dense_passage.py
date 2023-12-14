from transformers import DPRReader, AdamW, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os


class MarginLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Args:
            anchor: Tensor, embeddings for anchor samples.
            positive: Tensor, embeddings for positive samples.
            negative: Tensor, embeddings for negative samples.

        Returns:
            loss: Scalar tensor, the triplet loss.
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        # Triplet loss function
        loss = torch.mean(torch.relu(distance_positive - distance_negative + self.margin))

        return loss


def train_dpr_model(documents, model_name='facebook/dpr-reader-single-nq-base', epochs=10):
    # Load the pre-trained DPR model
    model = DPRReader.from_pretrained(model_name)

    # Load the DPR tokenizer
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)

    # Define loss function and optimizer
    loss_fn = MarginLoss()
    optimizer = AdamW(model.parameters())

    # Prepare training data (format your documents into pairs/triplets)
    for epoch in range(epochs):
        for doc in documents:
            # Access the "words" field in the document info
            words = doc["words"]

            # Tokenize the document
            inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)

            # Forward pass
            encoded_docs = model(**inputs)

            # Calculate loss based on your chosen function and data format
            loss = loss_fn(encoded_docs)

            # Backpropagate and update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), 'trained_dpr_model.pt')

    # Return document embeddings and model (if needed for inference)
    return encoded_docs, model


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
