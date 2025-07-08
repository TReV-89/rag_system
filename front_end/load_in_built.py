import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from rag_methods import (
    split_and_load_documents_in_built
)

load_dotenv("../.env")
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY")
)
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, "..", "chroma_database")
file_path = os.path.normpath(file_path)
client = chromadb.PersistentClient(path=file_path)
rag_collection = client.get_or_create_collection(
       name="rag_collection_user", embedding_function=google_ef)

split_and_load_documents_in_built(rag_collection)
