from pathlib import Path
from typing import List, Tuple
import pypdf
import re


class PDFParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"❌ Erreur lecture PDF {pdf_path}: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        # Supprimer multiples espaces/newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer caractères spéciaux problématiques
        text = text.replace('\x00', '')
        
        return text.strip()
    
    @staticmethod
    def split_into_chunks(
        text: str,
        chunk_size: int = 500,
        overlap: int = 100
    ) -> List[str]:
        # Diviser par phrases (approximatif)
        sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = sentence.split()
            
            if current_size + len(words) > chunk_size and current_chunk:
                # Créer un chunk
                chunk_text = '. '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Garder dernières phrases pour overlap
                overlap_words = ' '.join(words[-overlap//10:])
                current_chunk = [overlap_words]
                current_size = len(overlap_words.split())
            
            current_chunk.append(sentence)
            current_size += len(words)
        
        # Dernier chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    @staticmethod
    def parse_pdf_with_metadata(
        pdf_path: str,
        chunk_size: int = 500
    ) -> List[Tuple[str, Dict]]:
        # Extraire texte
        text = PDFParser.extract_text_from_pdf(pdf_path)
        text = PDFParser.clean_text(text)
        
        # Découper en chunks
        chunks = PDFParser.split_into_chunks(text, chunk_size=chunk_size)
        
        # Créer métadonnées
        pdf_name = Path(pdf_path).name
        results = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": pdf_name,
                "path": pdf_path,
                "chunk_id": i,
                "chunk_count": len(chunks)
            }
            results.append((chunk, metadata))
        
        return results


if __name__ == "__main__":
    # Test simple
    print("Module PDF Parser - OK")
    print("Utilisation: PDFParser.extract_text_from_pdf('file.pdf')")