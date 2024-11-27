import streamlit as st

with st.chat_message("user"):
    st.write("Hello Joseph 👋")

    prompt = st.chat_input("Say something")
if prompt:
    st.write(f"{prompt}")