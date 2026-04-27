"""
Script de test pour vérifier l'extraction de texte des PDFs
"""

from src.pdf_parser import PDFParser
from src.utils import get_pdf_files

print("🔍 Recherche des PDFs...")
pdf_files = get_pdf_files()

print(f"✅ {len(pdf_files)} PDFs trouvés\n")

for i, pdf_path in enumerate(pdf_files, 1):
    print(f"📄 [{i}] {pdf_path}")
    
    # Extraire texte
    text = PDFParser.extract_text_from_pdf(pdf_path)
    
    # Nettoyer
    text_clean = PDFParser.clean_text(text)
    
    # Découper en chunks
    chunks = PDFParser.split_into_chunks(text_clean, chunk_size=500)
    
    print(f"   Texte brut: {len(text)} caractères")
    print(f"   Texte nettoyé: {len(text_clean)} caractères")
    print(f"   Chunks: {len(chunks)} chunks")
    print(f"   Premier chunk: {chunks[0][:100]}...")
    print()