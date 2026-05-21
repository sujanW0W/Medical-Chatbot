import os
from pinecone import Pinecone
from pinecone import ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from src.helpers import download_embedding_model, text_split, load_pdf_file
from dotenv import load_dotenv

load_dotenv()

embedding_model = download_embedding_model()
extracted_pdf_data = load_pdf_file("data")
text_chunks = text_split(extracted_pdf_data)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medical-chatbot"
dimension_of_vector = 384

pc.create_index(
    name=index_name,
    dimension=dimension_of_vector,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Embed and upsert the embeddings into the Pinecone Index

doc_store = PineconeVectorStore.from_documents(
    index_name=index_name,
    embedding=embedding_model,
    documents=text_chunks
)
