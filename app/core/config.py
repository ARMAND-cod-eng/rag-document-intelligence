from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Document Intelligence API"
    APP_ENV: str = "development"

    # API Keys
    TOGETHER_API_KEY: str

    # Storage
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    UPLOAD_DIR: str = "./data/uploads"

    # RAG settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5

    # Embedding model — runs locally, no GPU needed
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # LLM
    LLM_MODEL: str = "meta-llama/Llama-3-70b-chat-hf"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    class Config:
        env_file = ".env"

settings = Settings()