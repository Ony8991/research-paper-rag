import os
from pathlib import Path
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

def get_config():
    config = {
        "data_path": os.getenv("DATA_PATH", "./data/documents"),
        "chroma_path": os.getenv("CHROMA_PATH", "./data/chroma_db"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "generation_model": os.getenv("GENERATION_MODEL", "mistral:7b-instruct"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
    }
    return config


def get_documents_path():
    config = get_config()
    return Path(config["data_path"])


def get_pdf_files():
    doc_path = get_documents_path()
    pdf_files = []
    
    # Récursif: cherche tous les PDFs
    for pdf in doc_path.rglob("*.pdf"):
        pdf_files.append(str(pdf))
    
    return sorted(pdf_files)


def ensure_directory_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test
    config = get_config()
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nPDFs trouvés:")
    pdfs = get_pdf_files()
    for pdf in pdfs:
        print(f"  - {pdf}")