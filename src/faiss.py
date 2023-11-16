from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DB_FAISS_PATH = 'vectorstore/db_faiss'


def create_vector_db(file_paths):
    documents = []

    # Load each file from the provided list
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)  # Create a new loader instance for each file
        loaded_doc = loader.load()
        if loaded_doc:  # Check if the document is loaded successfully
            documents.append(loaded_doc)

    # Rest of the process remains the same
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Ensuring directory exists
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

