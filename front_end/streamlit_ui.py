import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rag_methods import (
    clear_chat,
    initialize_session_states,
    generate_response,
    display_chat_messages,
    handle_file_upload,
    
)

# Load environment variables
load_dotenv("../.env")
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not in .env file.")

# Initialize ChromaDB
client = chromadb.HttpClient(host="chroma", port=8000)

# Check ChromaDB connection
if not (ret := client.heartbeat()):
    st.error(
        "ChromaDB server is not running. Please start the server before using this app."
    )
    st.stop()
else:
    st.write("ChromaDB server is running.")

# Initialize collection
if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = client.get_or_create_collection(
        name="rag_collection_user")

# Initialize LLM
if "llm" not in st.session_state:
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.7,
        max_tokens=None,
    )
    st.session_state.llm = llm
else:
    llm = st.session_state.llm

# Initialize session states
initialize_session_states()

# Streamlit Interface
st.title("RAG SYSTEM")

# Display chat messages
display_chat_messages()

# Handle chat input
if message := st.chat_input("Type your message here..."):
    st.session_state.messages.append(HumanMessage(content=message))
    response = generate_response(message, st.session_state.rag_collection, llm)
    st.rerun()

# Sidebar
with st.sidebar:
    if st.sidebar.button("🗑️ Clear Chat", type="primary"):
        clear_chat()
        st.sidebar.success("Chat cleared!")
        st.rerun()

    if file := st.file_uploader("Upload a file", type=["txt", "pdf", "docx"]):
        if handle_file_upload(file, st.session_state.rag_collection):
            st.rerun()
