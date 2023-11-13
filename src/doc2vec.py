import json
from gensim.models import Doc2Vec
from preprocess import load_data, preprocess_data


def train_doc2vec(tagged_data):
    model = Doc2Vec(vector_size=100,  # Increased vector size
                    window=5,         # Context window size
                    min_count=1,      # Ignore words with frequency less than 5
                    dm=1,             # Using PV-DM
                    epochs=100,       # More epochs
                    alpha=0.025,      # Initial learning rate
                    min_alpha=0.00025,# Minimum learning rate
                    sample=1e-5       # Downsample setting for frequent words
                   )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


if __name__ == "__main__":
    # Load preprocessed documents
    processed_docs = preprocess_data(load_data('../data/raw/translations.json'))

    # Train and save Doc2Vec model
    doc2vec_model = train_doc2vec(processed_docs)
    doc2vec_model.save("../models/doc2vec_model.bin")
