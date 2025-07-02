from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
import os
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from chromadb.utils import embedding_functions
import chromadb

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not in .env file.")

from langchain_openai import AzureOpenAIEmbeddings

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY")
)

client = chromadb.HttpClient(host="localhost", port=8000)  # Adjust the URL if needed
if not (ret := client.heartbeat()):
    st.error(
        "ChromaDB server is not running. Please start the server before using this app."
    )
    st.stop()
else:
    st.write("ChromaDB server is running.")

DB_DOCS_LIMIT = 5

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = client.delete_collection(name="rag_collection")
    st.session_state.rag_collection = client.get_or_create_collection(
        name="rag_collection", embedding_function=google_ef
    )


def split_and_load_documents(docs):
    chunker = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunked_docs = chunker.split_documents(docs)
    
    def documents_to_texts(docs):
        return [doc.page_content for doc in docs]

    def documents_to_metadatas(docs):
        return [doc.metadata for doc in docs
                ]
    texts = documents_to_texts(chunked_docs)
    metadatas = documents_to_metadatas(chunked_docs)    
        
    st.session_state.rag_collection.add(
        documents=texts, metadatas = metadatas, ids=[f"id{i}" for i in range(len(chunked_docs))]
    )


system_message = SystemMessage(
    content="You are a helpful assistant that is not verbose. Only answer questions based on the results from the documents provided. If you don't know the answer, say 'I don't know'."
)

# check if llm is already in session state
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

if "lastest_messages_sent" not in st.session_state:
    st.session_state.lastest_messages_sent = []

if "file_path" not in st.session_state:
    st.session_state.file_path = None


st.title("RAG SYSTEM")


if "messages" not in st.session_state:
    st.session_state.messages = []


def generate_response(messages: str):
    results = st.session_state.rag_collection.query(query_texts=[messages], include =["documents"], n_results=5)
    docs = results["documents"][0]
    combined_system_message = f"""{system_message} , Use the following documents to answer the question: {"\n".join(docs)}"""
        
    message = [
        HumanMessage(content=combined_system_message)
    ] + st.session_state.messages

    prompt = ChatPromptTemplate.from_messages([("human", "The user asked this : {input}, Use  the system message to answer the question: {message}")])
    chain = prompt | llm
    response = chain.invoke({"input": messages, "message": message})
    st.session_state.messages.append(response)
    st.session_state.lastest_messages_sent.append(response)
    return response


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if message := st.chat_input("Type your message here..."):
    st.session_state.messages.append(HumanMessage(content=message))
    response = generate_response(message)
    st.rerun()

with st.sidebar:
    if file := st.file_uploader("Upload a file", type=["txt", "pdf", "docx"]):
        rag_documents = []
        st.session_state.file_path = f"files_uploaded/{file.name}"
        if not os.path.exists(f"files_uploaded/{file.name}"):
            file.seek(0)  # Reset file pointer to the beginning
            contents = file.read()
            with open(f"files_uploaded/{file.name}", "wb") as f:
                f.write(contents)
        st.success(f"File '{file.name}' uploaded successfully!")
        try:
            if file.name.endswith(".txt"):
                loader = TextLoader(st.session_state.file_path, encoding="utf-8")
                rag_documents = loader.load()
                st.success("Documents loaded successfully!")
            elif file.name.endswith(".pdf"):
                loader = PyPDFLoader(st.session_state.file_path)
                rag_documents = loader.load()

            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(st.session_state.file_path)
                rag_documents = loader.load()

            else:
                st.error(f"Unsupported file type: {file.name}")

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

        if rag_documents:
            split_and_load_documents(rag_documents)
            st.success("Documents loaded successfully!")
