import os
import re
from typing import List, Dict

def count_words(text: str) -> int:
    """Retorna el número de palabras en un texto."""
    return len(text.split())

def extract_metadata_from_filename(filepath: str) -> Dict:
    """
    Extrae libro, chunk_size y overlap a partir del path.
    Ejemplo path:
        data/preprocessed/processed_400_100/plot_summary_twilight_chunks.txt
    """
    folder = os.path.basename(os.path.dirname(filepath))  # processed_400_100
    match = re.search(r"processed_(\d+)_(\d+)", folder)

    chunk_size = int(match.group(1)) if match else None
    overlap = int(match.group(2)) if match else None

    filename = os.path.basename(filepath).replace("_chunks.txt", "")
    book_name = filename.replace("plot_summary_", "")

    return {
        "book_name": book_name,
        "chunk_size": chunk_size,
        "overlap": overlap
    }

def load_chunks_from_folder(base_folder: str) -> List[Dict]:
    """
    Carga todos los chunks .txt desde carpetas processed_* y retorna una lista de dicts.
    Cada elemento tendrá: chunk_id, text, book_name, chunk_size, overlap, chunk_number, word_count
    """
    records = []
    chunk_id = 1

    for root, _, files in os.walk(base_folder):
        for file in files:
            if not file.endswith("_chunks.txt"):
                continue

            filepath = os.path.join(root, file)
            meta = extract_metadata_from_filename(filepath)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = content.split("--- CHUNK ")
            for item in chunks:
                if not item.strip():
                    continue

                number_match = re.match(r"(\d+) ---\n(.+)", item, re.DOTALL)
                if not number_match:
                    continue

                chunk_number = int(number_match.group(1))
                text = number_match.group(2).strip()
                word_count = count_words(text)

                records.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "book_name": meta["book_name"],
                    "chunk_size": meta["chunk_size"],
                    "overlap": meta["overlap"],
                    "chunk_number": chunk_number,
                    "word_count": word_count
                })
                chunk_id += 1

    return records