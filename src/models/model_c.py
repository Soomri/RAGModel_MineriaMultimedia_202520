import os
import glob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Importa funciones de chunking e indexación
from src.chunking import clean_text
from src.indexacion import create_tfidf_index, query_tfidf


class RAGModelC:
    """
    Modelo RAG con eliminación de redundancia entre chunks (Model C).
    """

    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = preprocessed_dir
        self.chunks = []
        self.vectorizer = None
        self.X = None

    # ------------------------------------------------------------
    # 🔹 1. Cargar documentos
    # ------------------------------------------------------------
    def load_documents(self):
        txt_files = glob.glob(os.path.join(self.preprocessed_dir, "*.txt"))
        print(f"📚 Se cargaron {len(txt_files)} archivos.")
        docs = []
        for path in txt_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    clean = clean_text(text)
                    docs.append(clean)
            except Exception as e:
                print(f"⚠️ Error leyendo {path}: {e}")
        return docs

    # ------------------------------------------------------------
    # 🔹 2. Dividir en chunks
    # ------------------------------------------------------------
    def chunk_documents(self, docs, chunk_size=300, overlap=50):
        all_chunks = []
        for doc in docs:
            words = doc.split()
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk = " ".join(words[start:end])
                all_chunks.append(chunk)
                start += chunk_size - overlap
        print(f"📊 Total de chunks antes de filtrar: {len(all_chunks)}")
        return all_chunks

    # ------------------------------------------------------------
    # 🔹 3. Eliminar redundancias
    # ------------------------------------------------------------
    def remove_redundant_chunks(self, chunks, similarity_threshold=0.9):
        print("🧩 Eliminando redundancias entre chunks...")
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        X = vectorizer.fit_transform(chunks)
        similarity_matrix = cosine_similarity(X)

        unique_chunks = []
        seen = set()

        for i in range(len(chunks)):
            if i in seen:
                continue
            duplicates = np.where(similarity_matrix[i] > similarity_threshold)[0]
            seen.update(duplicates)
            unique_chunks.append(chunks[i])

        print(f"✅ {len(unique_chunks)}/{len(chunks)} chunks únicos conservados.")
        return unique_chunks

    # ------------------------------------------------------------
    # 🔹 4. Preparar documentos e indexar
    # ------------------------------------------------------------
    def prepare_documents(self, docs):
        all_chunks = self.chunk_documents(docs)
        self.chunks = self.remove_redundant_chunks(all_chunks)
        self.vectorizer, self.X = create_tfidf_index(self.chunks)

    # ------------------------------------------------------------
    # 🔹 5. Consulta
    # ------------------------------------------------------------
    def query(self, query_text, top_k=3):
        print(f"\n🔎 Consulta: {query_text}\n")
        results = query_tfidf(query_text, self.vectorizer, self.X, self.chunks, top_k)

        print("🧩 Contextos recuperados:\n")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"[{i}] ({score:.3f}) {chunk[:300]}...\n")

        # Respuesta simulada (chunk más relevante)
        print("💬 Respuesta simulada:\n", results[0][0][:600])
        return results


if __name__ == "__main__":
    preprocessed_dir = r"C:\Users\USER\RAGModel_MineriaMultimedia_202520\data\preprocessed\processed_400_100"

    rag_c = RAGModelC(preprocessed_dir)
    docs = rag_c.load_documents()
    rag_c.prepare_documents(docs)

    # Ejemplo de consulta
    rag_c.query("who is bella?")
