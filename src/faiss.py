from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DB_FAISS_PATH = '../model/faiss_db'


# Create vector database
def create_vector_db(docs):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)

    # Ensuring directory exists
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()