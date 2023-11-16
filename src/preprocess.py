import json
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument


def preprocess_data_pdf_to_json(document, tags="file"):
    tagged_documents = []
    if tags == "":
        i = 0
        for entry in document:
            for data in entry:
                if data:
                    text = data
                    words = word_tokenize(text.lower())
                    tagged_documents.append(TaggedDocument(words=words, tags=[i]))
                    i = i+1
    else:
        i = 0
        for entry in document:
            for data in entry:
                if data:
                    text = data
                    words = word_tokenize(text.lower())
                    tagged_documents.append(TaggedDocument(words=words, tags=[tags + str(i)]))
                    i = i + 1
    return tagged_documents


def preprocess_data_qur(document):
    tagged_documents = []
    for entry in document:
        surah_id = entry['surah_id']
        verse_id = entry['verse_id']
        text = entry['text']
        tags = [f"surah_{surah_id}_verse_{verse_id}"]
        words = word_tokenize(text.lower())
        tagged_documents.append(TaggedDocument(words=words, tags=tags))
    return tagged_documents


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

