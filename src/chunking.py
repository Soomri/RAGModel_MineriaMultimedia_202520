import os
import re
import json
from typing import List

# chunking.py — Generación de chunks con solapamiento (.txt y .json)


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
        print(f"Error leyendo {filepath}: {e}")
        return ""


def process_documents(base_input: str, base_output: str, chunk_size: int = 400, overlap: int = 100):
    """
    Lee documentos (.txt y .json), limpia el texto y genera chunks.
    Crea una subcarpeta automática según el tamaño y el solapamiento de los chunks.
    """
    # Crea carpeta con nombre según configuración
    output_folder = os.path.join(base_output, f"processed_{chunk_size}_{overlap}")
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n Procesando documentos con chunk_size={chunk_size}, overlap={overlap}")
    print(f" Carpeta destino: {output_folder}\n")

    for filename in os.listdir(base_input):
        input_path = os.path.join(base_input, filename)
        if filename.endswith('.txt'):
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif filename.endswith('.json'):
            text = read_json_text(input_path)
        else:
            continue  # ignora otros tipos de archivo

        clean = clean_text(text)
        chunks = split_into_chunks(clean, chunk_size, overlap)

        # Guarda el resultado por documento
        output_name = filename.replace('.txt', '').replace('.json', '') + '_chunks.txt'
        output_path = os.path.join(output_folder, output_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f'--- CHUNK {i+1} ---\n{chunk}\n\n')

        print(f" {filename}: {len(chunks)} chunks creados.")

    print(f"\n Proceso completado para {chunk_size}-{overlap}. Archivos guardados en: {output_folder}\n")


if __name__ == "__main__":
    base_input = r"C:\Users\USER\RAGModel_MineriaMultimedia_202520\data\raw"
    base_output = r"C:\Users\USER\RAGModel_MineriaMultimedia_202520\data\preprocessed"

    # Puedes probar varios tamaños y solapamientos
    configs = [
        (400, 100),
        (600, 150),
        (800, 200),
    ]

    for chunk_size, overlap in configs:
        process_documents(base_input, base_output, chunk_size, overlap)