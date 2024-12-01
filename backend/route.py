from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import streamlit as st
import os
from openai import OpenAI
from langchain.schema import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)    # Connect to Pinecone
vectorstore = PineconeVectorStore(
            index_name="codebase-rag", 
            embedding=HuggingFaceEmbeddings(
                 model_name="sentence-transformers/all-mpnet-base-v2"
            ))

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
    ext = st.path.splitext(file_name)[1]
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

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)
    print("Query embedding shape:", raw_query_embedding.shape)

    top_matches = pc.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace="https://github.com/CoderAgent/SecureAgent")
    file_names = [item['metadata']['source'] for item in top_matches['matches']]

    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    context_summary = "\n\n".join(
        [f"File: {file_name}\nContent:\n{context[:500]}..." for file_name, context in zip(file_names, contexts)]
    )

    augmented_query = f"""
    # Codebase Context:
    {context_summary}

    # Developer Question:
    {query}

    Please provide a response based on the provided context and the specific question.
    """

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content