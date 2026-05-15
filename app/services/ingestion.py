from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.core.config import settings
import os

# Load embedding model once — reused for all documents
embeddings = SentenceTransformerEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

def get_vector_store():
    """Get or create the ChromaDB vector store."""
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )

def ingest_document(file_path: str, filename: str) -> dict:
    """
    Full pipeline: PDF → text → chunks → embeddings → ChromaDB
    Returns a summary of what was ingested.
    """

    # STEP 1 — Load PDF and extract text
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        raise ValueError(f"Could not extract text from {filename}")

    # STEP 2 — Split into chunks
    # Why RecursiveCharacterTextSplitter?
    # It tries to split on paragraphs first, then sentences, then words.
    # This keeps semantic meaning together better than splitting every N chars.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    # Add filename metadata to every chunk
    # This lets us cite the source in answers
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["filename"] = filename

    # STEP 3 — Embed and store in ChromaDB
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    vector_store.persist()

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "message": f"Successfully ingested {len(chunks)} chunks from {len(pages)} pages"
    }

def list_documents() -> list:
    """List all documents currently stored in ChromaDB."""
    vector_store = get_vector_store()
    collection = vector_store._collection
    results = collection.get()

    # Extract unique filenames from metadata
    filenames = set()
    for metadata in results.get("metadatas", []):
        if metadata and "filename" in metadata:
            filenames.add(metadata["filename"])

    return list(filenames)

def delete_document(filename: str) -> dict:
    """Delete all chunks belonging to a document."""
    vector_store = get_vector_store()
    collection = vector_store._collection

    # Find all chunk IDs for this document
    results = collection.get(where={"filename": filename})
    ids_to_delete = results.get("ids", [])

    if not ids_to_delete:
        raise ValueError(f"Document '{filename}' not found")

    collection.delete(ids=ids_to_delete)

    return {
        "filename": filename,
        "chunks_deleted": len(ids_to_delete),
        "message": f"Deleted {len(ids_to_delete)} chunks for '{filename}'"
    }