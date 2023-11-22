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

