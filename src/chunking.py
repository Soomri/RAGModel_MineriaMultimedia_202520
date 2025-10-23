import os
import re
import json
from typing import List

# 🧩 chunking.py — Generación de chunks con solapamiento (.txt y .json)


def clean_text(text: str) -> str:
    """Limpia el texto eliminando caracteres innecesarios y espacios extras."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """Divide el texto en chunks con solapamiento."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def read_json_text(filepath: str) -> str:
    """Lee un archivo JSON y devuelve el texto concatenado si tiene campos tipo texto."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Intentamos extraer texto de campos comunes o concatenar todo lo textual
        if isinstance(data, dict):
            text_parts = [str(v) for v in data.values() if isinstance(v, (str, int, float))]
            return ' '.join(text_parts)
        elif isinstance(data, list):
            text_parts = []
            for item in data:
                if isinstance(item, dict):
                    text_parts += [str(v) for v in item.values() if isinstance(v, (str, int, float))]
                elif isinstance(item, str):
                    text_parts.append(item)
            return ' '.join(text_parts)
        else:
            return str(data)
    except Exception as e:
        print(f"⚠️ Error leyendo {filepath}: {e}")
        return ""


def process_documents(input_folder: str, output_folder: str, chunk_size: int = 400, overlap: int = 100):
    """Lee documentos (.txt y .json), limpia el texto y genera chunks."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.endswith('.txt'):
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif filename.endswith('.json'):
            text = read_json_text(input_path)
        else:
            continue  # ignora otros tipos de archivo

        clean = clean_text(text)
        chunks = split_into_chunks(clean, chunk_size, overlap)

        output_name = filename.replace('.txt', '').replace('.json', '') + '_chunks.txt'
        output_path = os.path.join(output_folder, output_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f'--- CHUNK {i+1} ---\n{chunk}\n\n')

        print(f"✅ {filename}: {len(chunks)} chunks creados.")


if __name__ == "__main__":
    process_documents(
        input_folder=r"C:\Users\USER\RAGModel_MineriaMultimedia_202520\data\raw",
        output_folder=r"C:\Users\USER\RAGModel_MineriaMultimedia_202520\data\processed",
        chunk_size=400,
        overlap=100
    )
