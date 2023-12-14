import pickle
import umap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def train_tfidf(docs):
    # Extract words and tags from TaggedDocument objects
    words = [doc.words for doc in docs]
    tags = [doc.tags for doc in docs]
    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'text': words, 'tags': tags})

    vectorizer = TfidfVectorizer()

    # fit the vectorizer to the text and transform the data
    text_data = df['text'].apply(lambda x: ' '.join(x))
    X = vectorizer.fit_transform(text_data)

    # create UMAP embeddings for the posts
    reducer = umap.UMAP(n_neighbors=45,
                        n_components=2,
                        min_dist=0.1,
                        metric='cosine')

    umap_embeddings = reducer.fit_transform(X)

    with open('../models/tfidf/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    filename = '../models/tfidf/umap_model.sav'
    joblib.dump(reducer, filename)

    # Save the document vectors
    doc_vectors = umap_embeddings  # Assuming umap_embeddings are your document vectors
    doc_vectors_filename = '../models/tfidf/doc_vectors.pkl'
    os.makedirs(os.path.dirname(doc_vectors_filename), exist_ok=True)
    joblib.dump(doc_vectors, doc_vectors_filename)
