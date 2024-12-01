from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))    # Connect to Pinecone
vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
embeddings = HuggingFaceEmbeddings("sentence-transformers/paraphrase-xlm-r-multilingual-v1")    # Embed text
splitter = RecursiveCharacterTextSplitter() # Split text into characters

LANGUAGE_SPLITTER = {
    '.ts': Language.TS,
    '.js': Language.JS,
    '.py': Language.PYTHON,
    '.java': Language.JAVA,
    '.tsx': Language.TS,
    '.jsx': Language.JS,
    '.cpp': Language.CPP,
    '.swift': Language.SWIFT
}

def get_language_from_extension(file_name):
    ext = os.path.splitext(file_name)[1]
    return LANGUAGE_SPLITTER.get(ext)

def pinecone_feed(file_content, repo_url):
    documents = []

    def process_file(file):
            doc = Document(page_content=f"{file['name']}\n{file['content']}", metadata={"source": file['name']})
            documents.append(doc)
    for file in file_content:
        process_file(file)
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        index_name="codebase-rag",
        namespace=repo_url
    )