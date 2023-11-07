import json
from gensim.models import Doc2Vec
from preprocess import load_data, preprocess_data


def train_doc2vec(tagged_data):
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


if __name__ == "__main__":
    # Load preprocessed documents
    processed_docs = preprocess_data(load_data('../data/raw/translations.json'))

    # Train and save Doc2Vec model
    doc2vec_model = train_doc2vec(processed_docs)
    doc2vec_model.save("../models/doc2vec_model.bin")
