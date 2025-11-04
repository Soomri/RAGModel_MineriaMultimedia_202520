import sys
import os

# Agregar la raíz del proyecto al pythonpath
ROOT_DIR = os.path.abspath("..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

print(ROOT_DIR)

from src.chunking import clean_text, split_into_chunks, process_documents

BASE_INPUT = "../data/raw"
BASE_OUTPUT = "../data/preprocessed"

files = os.listdir(BASE_INPUT)
files

with open(os.path.join(BASE_INPUT, files[0]), 'r', encoding='utf-8') as f:
    print(f.read()[:800])

# Se cargaron 8 documentos crudos (7 TXT, 1 JSON).

raw_text = "Ejemplo con    espacios   y saltos de línea.\n\nOtro párrafo."
print("Antes:", raw_text)
print("Después:", clean_text(raw_text))

from src.chunking import split_into_chunks

sample_text = r"data\raw\plot_summary_breakingdawn_bookone.txt"
chunks = split_into_chunks(sample_text, chunk_size=400, overlap=100)
len(chunks), chunks[:2]

process_documents(BASE_INPUT, BASE_OUTPUT, chunk_size=400, overlap=100)

output_folder = os.path.join(BASE_OUTPUT, "processed_400_100")
os.listdir(output_folder)

with open(os.path.join(output_folder, os.listdir(output_folder)[0]), 'r', encoding='utf-8') as f:
    print(f.read()[:800])

"""
Finalmente,
* Se limpió el texto removiendo espacios y saltos innecesarios.
* Se generaron chunks con configuración (400, 100) produciendo 28 chunks.
* Se guardaron los chunks en:
> /data/preprocessed/processed_400_100/
"""