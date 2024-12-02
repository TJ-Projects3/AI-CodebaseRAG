from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from openai import OpenAI
from langchain.schema import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# Initialize Pinecone correctly
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# Create index if it doesn't exist
if "codebase-rag" not in pc.list_indexes().names():
    pc.create_index(
        name="codebase-rag",
        dimension=768,  # dimension for all-mpnet-base-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

# Get index using Index() method
index = pc.Index("codebase-rag")

vectorstore = PineconeVectorStore(
    index_name="codebase-rag", 
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
)

LANGUAGE_SPLITTER = {
    '.ts': Language.TS,
    '.js': Language.JS,
    '.py': Language.PYTHON,
    '.java': Language.JAVA,
    '.tsx': Language.TS,
    '.jsx': Language.JS,
    '.cpp': Language.CPP,
    '.swift': Language.SWIFT
}   # Define the language splitter

def get_language_from_extension(file_name):   # Function to get the language from the file extension
    ext = st.path.splitext(file_name)[1]    # Get the file extension
    return LANGUAGE_SPLITTER.get(ext)       # Return the language

# In backend/route.py - Update pinecone_feed function

from langchain.text_splitter import RecursiveCharacterTextSplitter

def pinecone_feed(file_content, repo_url):
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        documents = []
        for file in file_content:
            # Split content into chunks
            chunks = text_splitter.split_text(file['content'])
            
            # Create document for each chunk
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file['name'],
                        "chunk": i,
                        "text": chunk  # Store chunk text in metadata
                    }
                )
                documents.append(doc)
            
            print(f"Processed {file['name']}: {len(chunks)} chunks")
        
        if not documents:
            print("No documents to process")
            return False
            
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
            index_name="codebase-rag",
            namespace=repo_url
        )
        return True
        
    except Exception as e:
        print(f"Error in pinecone_feed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"): # Function to get the HuggingFace embeddings
    model = SentenceTransformer(model_name) # Create a SentenceTransformer object
    return model.encode(text) # Return the embeddings

client = OpenAI(   # Create an OpenAI object
    base_url="https://api.groq.com/openai/v1",  # Pass the base URL
    api_key=st.secrets["GROQ_API_KEY"]  # Pass the API key
)

def perform_rag(query, repo_url): # Function to perform RAG
    raw_query_embedding = get_huggingface_embeddings(query)     # Get the HuggingFace embeddings
    print("Query embedding shape:", raw_query_embedding.shape) # Print the shape of the query embedding

    top_matches = index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=str(repo_url)) # Get the top matches
    file_names = [item['metadata']['source'] for item in top_matches['matches']] # Get the file names

    contexts = [item['metadata']['text'] for item in top_matches['matches']] # Get the contexts
    context_summary = "\n\n".join( # Join the contexts
        [f"File: {file_name}\nContent:\n{context[:500]}..." for file_name, context in zip(file_names, contexts)] # Zip the file names and contexts
    )

    augmented_query = f"""
    # Codebase Context: 
    {context_summary} # Pass the context summary

    # Developer Question:
    {query} 

    Please provide a response based on the provided context and the specific question.
    """
    # ^^ This will augment the query with the context summary

    # RAG Prompt, will need prompt engineering
    system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    ) # Get the LLM response from the OpenAI API

    return llm_response.choices[0].message.content # Return the LLM response