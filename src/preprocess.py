import json
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument


def preprocess_data(documents):
    tagged_documents = []
    for entry in documents:
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


"""# Load the JSON data from file
with open('../data/raw/quran_en.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract translations with their IDs
translations = []
for surah in data:
    for verse in surah['verses']:
        translation_entry = {
            'surah_id': surah['id'],
            'verse_id': verse['id'],
            'text': verse['translation']
        }
        translations.append(translation_entry)
# Write translations to a file
with open('../data/raw/translations.json', 'w', encoding='utf-8') as file:
    json.dump(translations, file, ensure_ascii=False, indent=4)

# Print translations or write them to a file
for entry in translations:
    print(f"Surah ID: {entry['surah_id']}, Verse ID: {entry['verse_id']}, Translation: {entry['text']}")
"""


if __name__ == "__main__":
    # Load and preprocess documents
    documents = load_data('../data/raw/translations.json')
    processed_docs = preprocess_data(documents)

    # Save processed docs if necessary
    with open('../data/processed/processed_quran.json', 'w') as file:
        json.dump([doc.words for doc in processed_docs], file)
