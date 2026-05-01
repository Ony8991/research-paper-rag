import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict:
    return {
        "data_path": os.getenv("DATA_PATH", "./data/documents"),
        "vector_db_path": os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "generation_model": os.getenv("GENERATION_MODEL", "mistral:7b-instruct"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
    }


def get_documents_path() -> Path:
    return Path(get_config()["data_path"])


def get_pdf_files() -> list:
    doc_path = get_documents_path()
    return sorted(str(p) for p in doc_path.rglob("*.pdf"))


def ensure_directory_exists(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
