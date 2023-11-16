from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# BERT embedding and retrieval functions
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def encode_text_bert(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling


def retrieve_documents_bert(query, documents):
    query_embedding = encode_text_bert(query).detach().numpy()
    document_embeddings = [encode_text_bert(doc).detach().numpy() for doc in documents]
    similarities = cosine_similarity(query_embedding, document_embeddings)
    most_similar_idx = similarities[0].argsort()[-3:][::-1]  # Retrieve top 3 documents
    return [documents[idx] for idx in most_similar_idx]


