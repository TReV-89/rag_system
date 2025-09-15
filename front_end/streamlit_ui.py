import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from chromadb.utils import embedding_functions
from rag_methods import (
    clear_chat,
    initialize_session_states,
    generate_response,
    display_chat_messages,
    handle_file_upload,
)

load_dotenv("./.env")
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")
#google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
default_ef = embedding_functions.DefaultEmbeddingFunction()


if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not in .env file.")

chroma_host = os.getenv("CHROMA_HOST", "http://localhost:8000")

client = chromadb.HttpClient(host="chroma_host", port=443,ssl = True)

if not (ret := client.heartbeat()):
    st.error(
        "ChromaDB server is not running. Please start the server before using this app."
    )
    st.stop()
else:
    st.success("ChromaDB server is running.")

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = client.get_or_create_collection(
        name="rag_collection_user", embedding_function=default_ef
    )

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

initialize_session_states()


def get_document_count():
    try:
        collection = st.session_state.rag_collection
        return collection.count()
    except Exception as e:
        st.error(f"Error getting document count: {e}")
        return 0


def display_usage_guide():
    with st.expander("📖 How to Use the RAG System", expanded=False):
        st.markdown(
            """
        ### Welcome to the RAG (Retrieval-Augmented Generation) System!
        
        This system allows you to upload documents and ask questions about their content using AI.
        
        #### 🚀 Getting Started:
        
        **1. Upload Documents**
        - Use the file uploader in the sidebar
        - Supported formats: TXT, PDF
        - Upload multiple documents to build your knowledge base
        
        **2. Ask Questions**
        - Type your question in the chat input at the bottom
        - The system will search through your uploaded documents
        - Get AI-powered answers based on your document content
        
        #### 💡 Tips for Best Results:
        
        - **Be specific**: Ask clear, detailed questions
        - **Reference context**: Mention topics or document names when relevant
        - **Multiple documents**: Upload related documents for comprehensive answers
        - **Follow-up questions**: Build on previous responses for deeper insights
        
        #### 🔧 Features:
        
        - **Smart Search**: Finds relevant information across all uploaded documents
        - **Context-Aware**: Maintains conversation history for better responses
        - **Clear Chat**: Reset conversation anytime using the sidebar button
        - **Document Tracking**: See how many documents you've uploaded
        
       
        
        #### ⚠️ Important Notes:
        
        - Documents are processed and stored securely
        - Each upload adds to your searchable knowledge base
        - Clear chat only clears conversation, not uploaded documents
        - Larger documents may take a moment to process
        """
        )


st.title(" RAG SYSTEM")
st.markdown("*Retrieval-Augmented Generation for Document Q&A*")

display_usage_guide()

doc_count = get_document_count()
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("📚 Documents", doc_count)
with col2:
    if doc_count > 0:
        st.metric("🟢 Status", "Ready")
    else:
        st.metric("🟡 Status", "No Docs")

display_chat_messages()

if message := st.chat_input("Type your message here..."):
    if doc_count == 0:
        st.warning("⚠️ Please upload at least one document before asking questions.")
    else:
        st.session_state.messages.append(HumanMessage(content=message))
        response = generate_response(message, st.session_state.rag_collection, llm)
        st.rerun()

with st.sidebar:
    st.header(" Controls")

    st.subheader(" Database Status")
    st.info(f"Documents in database: **{doc_count}**")

    if doc_count > 0:
        st.success("✅ Ready to answer questions!")
    else:
        st.warning("⚠️ Upload documents to get started")

    if st.button("🗑️ Clear Chat", type="primary"):
        clear_chat()
        st.success("Chat cleared!")
        st.rerun()

    st.subheader("📁 Upload Documents")
    st.markdown("*Add documents to your knowledge base*")

    if file := st.file_uploader(
        "Choose files",
        type=["txt", "pdf"],
        help="Upload TXT or PDF files to add to your knowledge base",
    ):
        with st.spinner("Processing document..."):
            if handle_file_upload(file, st.session_state.rag_collection):
                st.success(f"✅ Successfully uploaded: {file.name}")
                st.rerun()
            else:
                st.error("❌ Failed to upload document")

    st.subheader("ℹ️ System Info")
    st.caption(f"Model: {model}")
    st.caption("ChromaDB: Connected")

    with st.expander("💡 Quick Tips"):
        st.markdown(
            """
        - Upload multiple related documents for better context
        - Ask specific questions for more accurate answers
        - Use follow-up questions to dive deeper
        - Clear chat to start fresh conversations
        """
        )
