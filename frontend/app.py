
import streamlit as st
import requests

st.set_page_config(page_title="Enterprise RAG System", layout="wide")
st.title("🚀 Enterprise RAG Query Interface")

query = st.text_input("Ask a question about your documents:")
if st.button("Submit Query"):
    if query:
        with st.spinner("Processing with LangChain & Vector DB..."):
            # Connecting to FastAPI backend
            try:
                response = requests.post("http://backend:8000/query", json={"text": query})
                st.write("### Response:")
                st.info(response.json().get("response", "No answer generated."))
            except Exception as e:
                st.error("Backend connection pending setup.")
