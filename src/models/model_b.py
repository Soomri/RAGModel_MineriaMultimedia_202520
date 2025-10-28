import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_chunks_from_folder

class RAGModelB:
    """
    Modelo RAG con chunking optimizado + deduplicación (Model B).
    Similar a C, pero con umbral más bajo y selección distinta de chunks representativos.
    """

    def __init__(self, preprocessed_base_dir=None, similarity_threshold=0.75):
        if preprocessed_base_dir is None:
            self.preprocessed_base_dir = os.path.join(os.getcwd(), "data", "preprocessed")
        else:
            self.preprocessed_base_dir = preprocessed_base_dir
            
        self.similarity_threshold = similarity_threshold
        self.chunks = []
        self.chunks_metadata = []
        self.vectorizer = None
        self.X = None
        self.original_count = 0
        self.deduplicated_count = 0

    # ---------------- Carga de chunks ----------------
    def load_preprocessed_chunks(self, chunk_config="processed_400_100"):
        folder_path = os.path.join(self.preprocessed_base_dir, chunk_config)
        records = load_chunks_from_folder(folder_path)
        df = pd.DataFrame.from_records(records)
        self.chunks = df['text'].astype(str).tolist()
        self.chunks_metadata = df.to_dict('records')
        self.original_count = len(self.chunks)
        return self.chunks

    # ---------------- Deduplicación ----------------
    def deduplicate_chunks(self):
        if not self.chunks:
            raise ValueError("No hay chunks cargados.")
        temp_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        temp_X = temp_vectorizer.fit_transform(self.chunks)
        similarity_matrix = cosine_similarity(temp_X)
        unique_chunks = []
        unique_metadata = []
        seen = set()
        for i in range(len(self.chunks)):
            if i in seen:
                continue
            similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
            seen.update(similar_indices)
            # Seleccionar chunk con MENOS palabras (para probar efecto)
            best_idx = min(similar_indices, key=lambda idx: len(self.chunks[idx].split()))
            unique_chunks.append(self.chunks[best_idx])
            unique_metadata.append(self.chunks_metadata[best_idx])
        self.chunks = unique_chunks
        self.chunks_metadata = unique_metadata
        self.deduplicated_count = len(self.chunks)
        return self.chunks

    # ---------------- Crear índice TF-IDF ----------------
    def create_index(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
        self.X = self.vectorizer.fit_transform(self.chunks)
        return self.vectorizer, self.X

    # ---------------- Pipeline completo ----------------
    def prepare_documents(self, chunk_config="processed_400_100"):
        self.load_preprocessed_chunks(chunk_config)
        self.deduplicate_chunks()
        self.create_index()
        print(f"Chunks originales: {self.original_count}")
        print(f"Chunks deduplicados: {self.deduplicated_count}")
        print(f"Features TF-IDF: {self.X.shape[1]}")

    # ---------------- Consulta ----------------
    def query(self, query_text, top_k=3):
        if self.vectorizer is None or self.X is None:
            raise ValueError("Índice no inicializado.")
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.X).flatten()
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.chunks[idx], similarities[idx], self.chunks_metadata[idx]) for idx in top_k_indices]
        return results

    # ---------------- Estadísticas ----------------
    def get_stats(self):
        return {
            "chunks_originales": self.original_count,
            "chunks_deduplicados": self.deduplicated_count,
            "reduccion_porcentaje": (1 - self.deduplicated_count/self.original_count)*100 if self.original_count>0 else 0,
            "features_tfidf": self.X.shape[1] if self.X is not None else 0,
            "umbral_similitud": self.similarity_threshold
        }

# ==============================
# EJEMPLO DE USO
# ==============================
if __name__ == "__main__":
    project_root = os.getcwd()
    preprocessed_base = os.path.join(project_root, "data", "preprocessed")
    
    rag_b = RAGModelB(preprocessed_base_dir=preprocessed_base, similarity_threshold=0.75)
    rag_b.prepare_documents(chunk_config="processed_400_100")
    
    queries = [
        "Who saves Bella from the van?",
        "Which Cullen family member is a doctor?",
        "Where does Bella move to?"
    ]
    
    for q in queries:
        results = rag_b.query(q, top_k=3)
        print(f"\nConsulta: {q}")
        for i, (chunk, score, _) in enumerate(results, 1):
            print(f"RANK {i} | Score: {score:.4f} | {chunk[:200]}...")
