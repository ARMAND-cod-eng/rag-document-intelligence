from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.core.config import settings

# Reuse the same embedding model
embeddings = SentenceTransformerEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

def get_vector_store():
    """Get the ChromaDB vector store."""
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )

def retrieve_relevant_chunks(question: str, filename: str = None) -> list:
    """
    Convert question to vector → find TOP_K most similar chunks in ChromaDB.
    
    If filename is provided → search only within that document.
    If filename is None → search across ALL documents.
    """
    vector_store = get_vector_store()

    # Build search filter
    search_filter = None
    if filename:
        search_filter = {"filename": filename}

    # Similarity search — this is the core of RAG
    # ChromaDB converts the question to a vector, then finds
    # the nearest vectors using cosine similarity
    results = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=settings.TOP_K_RESULTS,
        filter=search_filter
    )

    # Format results with content + metadata + relevance score
    chunks = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page", 0),
            "relevance_score": round(score, 4)
        })

    return chunks