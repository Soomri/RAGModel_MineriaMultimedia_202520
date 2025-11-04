# 02 - Indexación Unificada

# Objetivo: Crear índices TF-IDF, ChromaDB y FAISS a partir de chunks preprocesados.

# Flujo del notebook:
# 1. Cargar chunks desde data/preprocessed/processed_*
# 2. Crear índice TF-IDF (baseline)
# 3. Generar embeddings y poblar ChromaDB
# 4. Construir índice FAISS
# 5. Realizar consultas de prueba
# 6. Comparar resultados entre métodos

# ===================================
# 1. Configuración y Setup
# ===================================

import os
import sys
import pprint
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Configurar rutas del proyecto
project_root = os.path.abspath("..")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

print("Project root:", project_root)
print("Src path added:", src_path)

# -----------------------------------

from utils import load_chunks_from_folder

BASE_PREPROCESSED = os.path.join(project_root, "data", "preprocessed")
folders = sorted([
    os.path.join(BASE_PREPROCESSED, f) 
    for f in os.listdir(BASE_PREPROCESSED) 
    if f.startswith("processed_")
])

print("Carpetas procesadas detectadas:")
pprint.pprint(folders)

# ===================================
# 2. Carga de Datos y Metadata
# ===================================

records = []
for folder in folders:
    recs = load_chunks_from_folder(folder)
    print(f"Leídos {len(recs)} registros desde {folder}")
    records.extend(recs)

df = pd.DataFrame.from_records(records)
print(f"\n Total chunks cargados: {len(df)}")
df.head()

# -----------------------------------

def make_formatted_id(row):
    return f"{row['book_name']}_{row['chunk_size']}_{row['overlap']}_chunk_{int(row['chunk_number']):02d}"

df['formatted_chunk_id'] = df.apply(make_formatted_id, axis=1)

print("Ejemplo de formatted_chunk_id:")
display(df[['formatted_chunk_id', 'book_name', 'chunk_number', 'word_count']].head(6))

print("\nConteo por chunk_size:")
display(df['chunk_size'].value_counts())

print("\n Top libros por número de chunks:")
display(df['book_name'].value_counts())

# ===================================
# 3. Indexación TF-IDF (Baseline)
# ===================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_tfidf_index(chunks):
    if not chunks:
        raise ValueError("La lista de chunks está vacía. No se puede crear el índice TF-IDF.")
    
    print("Generando representaciones TF-IDF de los chunks...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(chunks)
    print(f"Indexados {len(chunks)} chunks.")
    return vectorizer, X

def query_tfidf(query, vectorizer, X, chunks, top_k=3):
    if not query.strip():
        raise ValueError("La consulta está vacía.")
    
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    top_chunks = [(chunks[i], similarities[i]) for i in top_indices]
    return top_chunks

# -----------------------------------

# IMPORTANTE: Guardar documentos para TF-IDF antes de cualquier reasignación
documents_tfidf = df['text'].astype(str).tolist()
vectorizer_tfidf, X_tfidf = create_tfidf_index(documents_tfidf)

print(f"\n Índice TF-IDF creado con {X_tfidf.shape[0]} documentos")
print(f"Dimensionalidad: {X_tfidf.shape[1]} features")

# ===================================
# 4. Indexación Vectorial (ChromaDB + FAISS)
# ===================================

from sentence_transformers import SentenceTransformer

embed_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
print(f"Cargando modelo de embeddings: {embed_model_name}")
embed_model = SentenceTransformer(embed_model_name)

def token_count(text):
    tokens = embed_model.tokenize(text)
    return tokens['input_ids'].shape[1]

df['token_count_sample'] = df['text'].apply(
    lambda x: token_count(x) if len(x.split()) < 1000 else None
)

print("\n Información de tokenización:")
display(df[['formatted_chunk_id', 'word_count', 'token_count_sample']].head(6))

# -----------------------------------

ids = df['formatted_chunk_id'].astype(str).tolist()
documents = df['text'].astype(str).tolist()
metadatas = df.apply(lambda r: {
    "book_name": r['book_name'],
    "chunk_size": int(r['chunk_size']) if pd.notnull(r['chunk_size']) else None,
    "overlap": int(r['overlap']) if pd.notnull(r['overlap']) else None,
    "chunk_number": int(r['chunk_number']),
    "word_count": int(r['word_count'])
}, axis=1).tolist()

print(f"Preparados {len(ids)} documentos para indexación")