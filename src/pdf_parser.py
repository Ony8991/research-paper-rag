from pathlib import Path
from typing import List, Tuple, Dict
import pypdf
import re


class PDFParser:

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\x00", "")
        return text.strip()

    @staticmethod
    def split_into_chunks(
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> List[str]:
        sentences = text.split(". ")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for sentence in sentences:
            words = sentence.split()
            if current_size + len(words) > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk))
                # Keep a small overlap window from the end of the previous chunk
                overlap_words = " ".join(words[-(overlap // 10):])
                current_chunk = [overlap_words]
                current_size = len(overlap_words.split())
            current_chunk.append(sentence)
            current_size += len(words)

        if current_chunk:
            chunks.append(". ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    @staticmethod
    def parse_pdf_with_metadata(
        pdf_path: str,
        chunk_size: int = 500,
    ) -> List[Tuple[str, Dict]]:
        text = PDFParser.extract_text_from_pdf(pdf_path)
        text = PDFParser.clean_text(text)
        chunks = PDFParser.split_into_chunks(text, chunk_size=chunk_size)

        pdf_name = Path(pdf_path).name
        return [
            (
                chunk,
                {
                    "source": pdf_name,
                    "path": pdf_path,
                    "chunk_id": i,
                    "chunk_count": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]
