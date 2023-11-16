from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


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


# Assume `documents` is a list of strings representing the content of each document
def retrieve_documents_doc2vec(model_choice, query_vector, documents, topn=5):
    if model_choice == "doc":
        model = Doc2Vec.load("../models/doc2vec/doc2vec_model.bin")
        doc_vectors = model.dv.vectors

        similarities = cosine_similarity([query_vector], doc_vectors).flatten()
        related_doc_indices = similarities.argsort()[-topn:][::-1]

        # Fetch the actual documents using the indices
        related_documents = [(idx, documents[idx]) for idx in related_doc_indices]
        print(related_documents)
        return related_documents
