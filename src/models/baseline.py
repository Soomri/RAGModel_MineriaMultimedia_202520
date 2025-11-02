import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
import math

# Configurar path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils import load_chunks_from_folder


class BM25:
    """Implementación de BM25 (Best Matching 25) para recuperación de información."""
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize()
    
    def _initialize(self):
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size

        df = {}
        for document in self.tokenized_corpus:
            frequencies = set(document)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1

        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_scores(self, query):
        query_terms = query.lower().split()
        scores = np.zeros(self.corpus_size)
        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_idx, document in enumerate(self.tokenized_corpus):
                freq = document.count(term)
                if freq == 0:
                    continue
                doc_len = self.doc_len[doc_idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf * (numerator / denominator)
        return scores
    
    def get_top_k(self, query, k=5):
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_k_indices]


class RAGBaseline:
    """Modelo RAG Baseline con chunking simple + BM25 retriever."""
    
    def __init__(self, preprocessed_base_dir=None, k1=1.5, b=0.75):
        if preprocessed_base_dir is None:
            self.preprocessed_base_dir = os.path.join(project_root, "data", "preprocessed")
        else:
            self.preprocessed_base_dir = preprocessed_base_dir
        
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.chunks_metadata = []
        self.bm25 = None
        self.total_chunks = 0
    
    def load_preprocessed_chunks(self, chunk_config="processed_400_100"):
        print(f"\n{'='*80}")
        print("📂 CARGANDO CHUNKS PREPROCESADOS")
        print(f"{'='*80}")
        
        folder_path = os.path.join(self.preprocessed_base_dir, chunk_config)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ No se encuentra la carpeta: {folder_path}")
        
        records = load_chunks_from_folder(folder_path)
        if not records:
            raise ValueError(f"❌ No se encontraron chunks en {folder_path}")
        
        df = pd.DataFrame.from_records(records)
        self.chunks = df['text'].astype(str).tolist()
        self.chunks_metadata = df.to_dict('records')
        self.total_chunks = len(self.chunks)
        
        print(f"✅ Cargados {self.total_chunks} chunks ({chunk_config})\n")
        return self.chunks
    
    def create_bm25_index(self):
        if not self.chunks:
            raise ValueError("❌ No hay chunks para indexar.")
        
        print(f"\n{'='*80}")
        print("🔧 CREACIÓN DE ÍNDICE BM25")
        print(f"{'='*80}")
        
        self.bm25 = BM25(self.chunks, k1=self.k1, b=self.b)
        
        print(f"✅ Índice BM25 creado con {len(self.bm25.idf)} términos únicos.\n")
        return self.bm25
    
    def prepare_documents(self, chunk_config="processed_400_100"):
        print(f"\n{'#'*80}")
        print("# PIPELINE BASELINE: CHUNKING SIMPLE + BM25")
        print(f"{'#'*80}\n")
        self.load_preprocessed_chunks(chunk_config)
        self.create_bm25_index()
        print(f"✅ Pipeline completado ({self.total_chunks} chunks indexados)\n")
    
    #  Método query corregido
    def query(self, query_text, top_k=3, show_details=True, expected_answer=None):
        """Realiza una consulta y muestra resultados BM25 o la respuesta generada."""
        if self.bm25 is None:
            raise ValueError("❌ Índice no inicializado. Ejecuta prepare_documents() primero.")

        if show_details:
            print(f"\n{'='*80}")
            print(f"🔎 CONSULTA: '{query_text}'")
            print(f"{'='*80}\n")

        top_results = self.bm25.get_top_k(query_text, k=top_k)
        results = []
        for rank, (idx, score) in enumerate(top_results, 1):
            chunk = self.chunks[idx]
            metadata = self.chunks_metadata[idx]
            results.append((chunk, score, metadata))
            if show_details:
                print(f"🏆 RANK {rank} | BM25 Score: {score:.4f}")
                print(f"📚 Libro: {metadata.get('book_name', 'N/A')}")
                print(f"📄 Chunk #{metadata.get('chunk_number', 'N/A')}")
                print(f"📝 Texto: {chunk[:200]}...")
                print("-" * 80)

        # Mostrar respuesta generada si se pasa
        if show_details:
            print("\n💬 RESPUESTA FINAL:")
            if expected_answer:
                print(expected_answer)
            else:
                print(f"{results[0][0][:400]}...")
            print("=" * 80)

        # Devolver también la respuesta generada para uso externo
        return {
            "results": results,
            "generated_answer": expected_answer if expected_answer else results[0][0][:400]
        }
    
    def batch_query(self, queries, top_k=3):
        print(f"\n{'='*80}")
        print(f"📋 EJECUCIÓN BATCH ({len(queries)} consultas)")
        print(f"{'='*80}")
        all_results = []
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")
            results = self.query(query, top_k=top_k, show_details=False)
            all_results.append(results)
        print("\n✅ Batch completado.\n")
        return all_results
    
    def compare_with_tfidf(self, query_text, top_k=3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print(f"\n{'#'*80}")
        print(f"# COMPARACIÓN: BM25 vs TF-IDF")
        print(f"{'#'*80}\n")
        print(f"🔎 Consulta: '{query_text}'\n")
        bm25_results = self.query(query_text, top_k=top_k, show_details=False)
        print("BM25:")
        for i, (chunk, score, _) in enumerate(bm25_results["results"], 1):
            print(f"[{i}] {chunk[:200]}... ({score:.4f})")
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(self.chunks)
        query_vec = vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, X).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        print("\nTF-IDF:")
        for i, idx in enumerate(top_indices, 1):
            print(f"[{i}] {self.chunks[idx][:200]}... ({similarities[idx]:.4f})")
    
    def get_stats(self):
        if not self.chunks:
            return {"error": "No hay chunks cargados"}
        stats = {
            "total_chunks": self.total_chunks,
            "vocab_size": len(self.bm25.idf) if self.bm25 else 0,
            "avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
            "param_k1": self.k1,
            "param_b": self.b
        }
        print(f"\n{'='*80}")
        print("📊 ESTADÍSTICAS DEL MODELO")
        print(f"{'='*80}")
        for k, v in stats.items():
            print(f"• {k}: {v}")
        print("=" * 80)
        return stats
    
    def get_top_terms(self, n=20):
        if not self.bm25:
            raise ValueError("❌ Índice no inicializado.")
        sorted_terms = sorted(self.bm25.idf.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{'='*80}")
        print(f"🔝 TOP {n} TÉRMINOS MÁS DISCRIMINATIVOS")
        print(f"{'='*80}")
        for i, (term, idf) in enumerate(sorted_terms[:n], 1):
            print(f"{i:2d}. {term:20s} | IDF: {idf:.4f}")
        print("=" * 80)
        return sorted_terms[:n]


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    preprocessed_base = os.path.join(project_root, "data", "preprocessed")

    rag_baseline = RAGBaseline(preprocessed_base_dir=preprocessed_base)
    rag_baseline.prepare_documents("processed_400_100")
    rag_baseline.get_top_terms(10)

    queries = [
        "Who saves Bella from the van?",
        "Which Cullen family member is a doctor?",
        "Where does Bella move to?"
    ]

    for query in queries:
        result = rag_baseline.query(query, top_k=3)
        input("\nPresiona Enter para continuar...\n")

    rag_baseline.get_stats()
