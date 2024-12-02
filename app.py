import streamlit as st
from backend.repo import process_repo
from backend.route import perform_rag

# Page config
st.set_page_config(
    page_title="Codebase RAG Bot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repo_processed" not in st.session_state:
    st.session_state.repo_processed = False

# Title and description
st.title(":globe_with_meridians: Codebase RAG Bot")
st.write("Chat with your codebase using AI! Enter a GitHub repository URL to begin.")

# Repository URL input
repo_url = st.text_input("Enter GitHub Repository URL:", 
                        placeholder="https://github.com/username/repository")

# Process repository button
if repo_url and not st.session_state.repo_processed:
    if st.button("Process Repository"):
        with st.spinner("Processing repository..."):
            try:
                success = process_repo(repo_url)
                if success:
                    st.session_state.repo_processed = True
                    st.success("Repository processed successfully!")
                else:
                    st.error("Failed to process repository.")
            except Exception as e:
                st.error(f"Error processing repository: {str(e)}")

# Chat interface
if st.session_state.repo_processed:
    st.markdown("---")
    st.subheader("Chat with your codebase")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your codebase..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        with st.spinner("Thinking..."):
            try:
                response = perform_rag(prompt, repo_url)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please process a repository to start chatting!")

# Add repository reset button
if st.session_state.repo_processed:
    if st.button("Process New Repository"):
        st.session_state.repo_processed = False
        st.session_state.messages = []
        st.rerun()