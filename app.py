import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="Enterprise AI Chat", page_icon="🤖")
st.title("Enterprise AI Assistant 🚀")
st.caption("Powered by Llama-3 & FastAPI Backend")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box at the bottom
if prompt := st.chat_input("Ask me anything about tech, coding, or RAG..."):
    # 1. Add user message to UI
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the FastAPI Backend
    with st.spinner("Processing through AI Engine..."):
        try:
            # Backend API URL
            API_URL = "http://backend:8000/api/chat"
            
            # Send POST request
            response = requests.post(API_URL, json={"query": prompt})
            
            # Check if request was successful
            if response.status_code == 200:
                ai_reply = response.json()["response"]
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            else:
                st.error(f"Backend Error: Received status code {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Failed! Please check if your FastAPI backend is running.")