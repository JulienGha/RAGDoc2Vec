import json
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument


def preprocess_data(documents):
    return [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(documents)]


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Load the JSON data from file
with open('data/quran_en.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract translations with their IDs
translations = []
for surah in data:
    for verse in surah['verses']:
        translation_entry = {
            'surah_id': surah['id'],
            'verse_id': verse['id'],
            'translation': verse['translation']
        }
        translations.append(translation_entry)
# Write translations to a file
with open('data/translations.json', 'w', encoding='utf-8') as file:
    json.dump(translations, file, ensure_ascii=False, indent=4)

# Print translations or write them to a file
for entry in translations:
    print(f"Surah ID: {entry['surah_id']}, Verse ID: {entry['verse_id']}, Translation: {entry['translation']}")


if __name__ == "__main__":
    # Load and preprocess documents
    documents = load_data('../data/documents.json')
    processed_docs = preprocess_data(documents)

    # Save processed docs if necessary
    with open('../data/processed_docs.json', 'w') as file:
        json.dump([doc.words for doc in processed_docs], file)
