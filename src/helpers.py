from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

"""
    This function loads the pdf file from the data directory
    and returns the documents
"""


def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


"""
    This function splits the documents into chunks
    and returns the text chunks
"""


def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


"""
    This function downloads the embedding model
    and returns the embedding model
"""


def download_embedding_model():
    """
    Downloads and returns both the Langchain embedding interface and the raw SentenceTransformer model.
    """

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    langchain_embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    raw_embedding_model = SentenceTransformer(model_name)

    return langchain_embeddings, raw_embedding_model
