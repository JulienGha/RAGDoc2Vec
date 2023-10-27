import gensim

# Load your text data
text_data = []
with open("my_text_data.txt", "r") as f:
    for line in f:
        text_data.append(line)

# Create a Doc2Vec model
model = gensim.models.Doc2Vec(text_data, vector_size=128, window=8, min_count=5)

# Save the model
model.save("my_doc2vec_model.bin")

import faiss

# Load the Doc2Vec model
model = gensim.models.Doc2Vec.load("my_doc2vec_model.bin")

# Create a FAISS index of the Doc2Vec embeddings
index = faiss.IndexFlatIP(128)
index.add(model.docvecs.vectors_norm)

# Save the index
index.save("my_rag_retriever.idx")
