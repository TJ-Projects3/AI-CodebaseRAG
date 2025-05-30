## Welcome to our AI Codebase RAG project!
Built by Tiwalayo and Oluwagbenro

# What does it do?
ThE purpose of this application is to make code searching & analysis from your Github repo easier, without having to open your IDE. The AI is designed to find pieces of code quickly and give tailored tips to understand,
and even improve code!

# How does it work?
It is built to ingest, embed, query, and respond. First, the user is able to copy and paste their GitHub repo link, We parse the github link for the repo name, clone the repo via the GitHub API, and extract all the main files of the codebase. These extracted files are then broken up by chunking (for better quality inference & responses) and embedded with Hugging Face sentence transformers model and indexed into a Pinecone vector store. This vector store is then leveraged by the AI model as a knowledge base and the frontend enables you to chat with your codebase
