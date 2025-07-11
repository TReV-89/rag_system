import os
import streamlit as st
import time
import pdfplumber
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def clear_chat():
    if "messages" in st.session_state:
        st.session_state.messages.clear()
    if "latest_messages_sent" in st.session_state:
        st.session_state.latest_messages_sent.clear()


def initialize_session_states():
    if "latest_messages_sent" not in st.session_state:
        st.session_state.latest_messages_sent = []

    if "file_path" not in st.session_state:
        st.session_state.file_path = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def documents_to_texts(docs):
    return [doc.page_content for doc in docs]


def documents_to_metadatas(docs):
    return [doc.metadata for doc in docs]


def split_and_load_documents(docs, collection):
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Splitting documents into chunks...")
    progress_bar.progress(20)
    time.sleep(3)
    chunker = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunked_docs = chunker.split_documents(docs)

    progress_bar.progress(40)
    status_text.text("Processing document chunks...")
    time.sleep(3)

    texts = documents_to_texts(chunked_docs)
    metadatas = documents_to_metadatas(chunked_docs)

    progress_bar.progress(60)
    status_text.text("Generating embeddings and adding to database...")
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=[f"id{i}" for i in range(len(chunked_docs))],
    )

    progress_bar.progress(100)
    status_text.text("✅ Documents successfully processed and added to database!")

    time.sleep(3)
    progress_bar.empty()
    status_text.empty()


def load_document(file_path, file_name):
    try:
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        elif file_name.endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
            return loader.load()
        else:
            st.error(f"Unsupported file type: {file_name}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None


def save_uploaded_file(file, upload_dir="files_uploaded"):

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file_path = os.path.join(upload_dir, file.name)

    if not os.path.exists(file_path):
        file.seek(0)  # Reset file pointer to the beginning
        contents = file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

    return file_path


def generate_response(message, collection, llm):

    # Query the collection for relevant documents
    results = collection.query(query_texts=[message], n_results=5)
    smallest_distance = results["distances"][0][0]
    if smallest_distance < 0.7:
        docs = results["documents"][0]

        system_message = SystemMessage(
            content="You are a helpful assistant. Only answer questions based on the results from the documents provided. If the answer is not from the document provided , say 'I can only answer questions about the document provided.'."
        )

        enhanced_system_message = f"""{system_message.content} , Use the following documents to answer the question: {"\n".join(docs)}"""

        messages = [
            SystemMessage(content=enhanced_system_message),
            HumanMessage(content=message),
        ]

        recent_messages = st.session_state.messages
        messages.extend(recent_messages)

        response = llm.invoke(messages)
        st.session_state.messages.append(response)
        st.session_state.latest_messages_sent.append(response)
        return response
    else:
        system_message = SystemMessage(
            content="You are a helpful assistant. Reply the user as you would normally do, but do not use any documents to answer the question."
        )
        messages = [system_message, HumanMessage(content=message)]
        recent_messages = st.session_state.messages
        messages.extend(recent_messages)
        response = llm.invoke(messages)
        st.session_state.messages.append(response)
        return response


def display_chat_messages():
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)


def handle_file_upload(file, collection):
    try:
        # Save file
        file_path = save_uploaded_file(file)
        st.session_state.file_path = file_path
        st.success(f"File '{file.name}' uploaded successfully!")

        # Load documents
        rag_documents = load_document(file_path, file.name)

        if rag_documents:
            # Split and load into collection
            split_and_load_documents(rag_documents, collection)
            st.success("Documents loaded successfully!")

    except Exception as e:
        st.error(f"Error processing file upload: {e}")
