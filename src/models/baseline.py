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
    """
    Implementación de BM25 (Best Matching 25) para recuperación de información.
    
    BM25 es un algoritmo de ranking basado en el modelo probabilístico de recuperación.
    Mejora TF-IDF al incorporar saturación de términos y longitud de documento.
    """
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        Inicializa BM25.
        
        Parámetros:
        - corpus: Lista de documentos (strings)
        - k1: Parámetro de saturación de frecuencia de términos (típicamente 1.2-2.0)
        - b: Parámetro de normalización por longitud (0-1, típicamente 0.75)
        """
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
        """Calcula estadísticas del corpus necesarias para BM25."""
        # Tokenizar documentos
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        
        # Calcular longitud promedio de documentos
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size
        
        # Calcular frecuencias de documentos
        df = {}
        for document in self.tokenized_corpus:
            frequencies = set(document)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1
        
        # Calcular IDF para cada término
        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_scores(self, query):
        """
        Calcula scores BM25 para una consulta.
        
        Parámetros:
        - query: String de consulta
        
        Retorna:
        - Array numpy con scores para cada documento
        """
        query_terms = query.lower().split()
        scores = np.zeros(self.corpus_size)
        
        for term in query_terms:
            if term not in self.idf:
                continue
            
            idf = self.idf[term]
            
            for doc_idx, document in enumerate(self.tokenized_corpus):
                # Frecuencia del término en el documento
                freq = document.count(term)
                
                if freq == 0:
                    continue
                
                # Longitud del documento
                doc_len = self.doc_len[doc_idx]
                
                # Fórmula BM25
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf * (numerator / denominator)
        
        return scores
    
    def get_top_k(self, query, k=5):
        """
        Retorna los top-k documentos más relevantes.
        
        Parámetros:
        - query: String de consulta
        - k: Número de resultados
        
        Retorna:
        - Lista de tuplas (índice, score)
        """
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_k_indices]


class RAGBaseline:
    """
    Modelo RAG Baseline con chunking simple + BM25 retriever.
    
    Pipeline:
    1. Cargar chunks preprocesados (sin modificaciones)
    2. Indexar con BM25 (método clásico de recuperación)
    3. Realizar consultas
    
    Este es el modelo más simple y sirve como baseline de comparación.
    """
    
    def __init__(self, preprocessed_base_dir=None, k1=1.5, b=0.75):
        """
        Inicializa el modelo baseline.
        
        Parámetros:
        - preprocessed_base_dir: Ruta base a data/preprocessed
        - k1: Parámetro BM25 de saturación (default: 1.5)
        - b: Parámetro BM25 de normalización (default: 0.75)
        """
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
    
    # ------------------------------------------------------------
    # 1. Cargar chunks preprocesados
    # ------------------------------------------------------------
    def load_preprocessed_chunks(self, chunk_config="processed_400_100"):
        """
        Carga chunks desde las carpetas preprocesadas (sin modificaciones).
        
        Parámetros:
        - chunk_config: Nombre de la carpeta (ej: "processed_400_100")
        """
        print(f"\n{'='*80}")
        print(f"📂 CARGANDO CHUNKS PREPROCESADOS")
        print(f"{'='*80}")
        
        folder_path = os.path.join(self.preprocessed_base_dir, chunk_config)
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ No se encuentra la carpeta: {folder_path}")
        
        print(f"📁 Carpeta: {folder_path}")
        
        # Cargar usando la función de utils
        records = load_chunks_from_folder(folder_path)
        
        if not records:
            raise ValueError(f"❌ No se encontraron chunks en {folder_path}")
        
        df = pd.DataFrame.from_records(records)
        
        # Extraer textos y metadata (sin procesamiento adicional)
        self.chunks = df['text'].astype(str).tolist()
        self.chunks_metadata = df.to_dict('records')
        self.total_chunks = len(self.chunks)
        
        print(f"✅ Cargados {self.total_chunks} chunks")
        print(f"📊 Configuración: {chunk_config}")
        print(f"⚙️ Chunking: SIMPLE (sin modificaciones)")
        print(f"{'='*80}\n")
        
        return self.chunks
    
    # ------------------------------------------------------------
    # 2. Crear índice BM25
    # ------------------------------------------------------------
    def create_bm25_index(self):
        """
        Crea el índice BM25 para recuperación.
        """
        if not self.chunks:
            raise ValueError("❌ No hay chunks para indexar.")
        
        print(f"\n{'='*80}")
        print(f"🔧 CREACIÓN DE ÍNDICE BM25")
        print(f"{'='*80}")
        print(f"📊 Documentos: {len(self.chunks)}")
        print(f"⚙️ Parámetros BM25:")
        print(f"   • k1 (saturación): {self.k1}")
        print(f"   • b (normalización): {self.b}")
        
        # Crear índice BM25
        self.bm25 = BM25(self.chunks, k1=self.k1, b=self.b)
        
        print(f"✅ Índice BM25 creado")
        print(f"📊 Vocabulario: {len(self.bm25.idf)} términos únicos")
        print(f"📏 Longitud promedio doc: {self.bm25.avgdl:.1f} palabras")
        print(f"{'='*80}\n")
        
        return self.bm25
    
    # ------------------------------------------------------------
    # 3. Pipeline completo: cargar + indexar
    # ------------------------------------------------------------
    def prepare_documents(self, chunk_config="processed_400_100"):
        """
        Pipeline completo baseline: carga chunks e indexa con BM25.
        
        Parámetros:
        - chunk_config: Configuración de chunks a usar
        """
        print(f"\n{'#'*80}")
        print(f"# PIPELINE BASELINE: CHUNKING SIMPLE + BM25")
        print(f"{'#'*80}\n")
        
        # Paso 1: Cargar chunks preprocesados (sin modificaciones)
        self.load_preprocessed_chunks(chunk_config)
        
        # Paso 2: Crear índice BM25
        self.create_bm25_index()
        
        print(f"\n{'#'*80}")
        print(f"# ✅ PIPELINE COMPLETADO")
        print(f"{'#'*80}")
        print(f"📊 Resumen:")
        print(f"  • Total chunks:        {self.total_chunks}")
        print(f"  • Método chunking:     SIMPLE")
        print(f"  • Método retrieval:    BM25")
        print(f"  • Vocabulario:         {len(self.bm25.idf)} términos")
        print(f"{'#'*80}\n")
    
    # ------------------------------------------------------------
    # 4. Consulta
    # ------------------------------------------------------------
    def query(self, query_text, top_k=3, show_details=True):
        """
        Realiza una consulta usando BM25.
        
        Parámetros:
        - query_text: Texto de la consulta
        - top_k: Número de resultados a retornar
        - show_details: Mostrar detalles de los resultados
        
        Retorna:
        - Lista de tuplas (chunk, score, metadata)
        """
        if self.bm25 is None:
            raise ValueError("❌ Índice no inicializado. Ejecuta prepare_documents() primero.")
        
        if show_details:
            print(f"\n{'='*80}")
            print(f"🔎 CONSULTA: '{query_text}'")
            print(f"{'='*80}\n")
        
        # Obtener scores BM25
        top_results = self.bm25.get_top_k(query_text, k=top_k)
        
        # Preparar resultados
        results = []
        for rank, (idx, score) in enumerate(top_results, 1):
            chunk = self.chunks[idx]
            metadata = self.chunks_metadata[idx]
            
            results.append((chunk, score, metadata))
            
            if show_details:
                print(f"🏆 RANK {rank} | BM25 Score: {score:.4f}")
                print(f"📚 Libro: {metadata.get('book_name', 'N/A')}")
                print(f"📄 Chunk #{metadata.get('chunk_number', 'N/A')}")
                print(f"📏 Palabras: {metadata.get('word_count', 'N/A')}")
                print(f"📝 Texto: {chunk[:300]}...")
                print("-" * 80)
        
        if show_details and results:
            print(f"\n💬 RESPUESTA BASADA EN CONTEXTO MÁS RELEVANTE:")
            print(f"{results[0][0][:500]}...")
            print("=" * 80 + "\n")
        
        return results
    
    # ------------------------------------------------------------
    # 5. Búsqueda batch (múltiples queries)
    # ------------------------------------------------------------
    def batch_query(self, queries, top_k=3):
        """
        Ejecuta múltiples consultas de forma eficiente.
        
        Parámetros:
        - queries: Lista de strings de consultas
        - top_k: Número de resultados por consulta
        
        Retorna:
        - Lista de resultados por cada consulta
        """
        print(f"\n{'='*80}")
        print(f"📋 EJECUCIÓN BATCH: {len(queries)} consultas")
        print(f"{'='*80}\n")
        
        all_results = []
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")
            results = self.query(query, top_k=top_k, show_details=False)
            all_results.append(results)
            
            # Mostrar solo el top-1
            if results:
                print(f"  → Top-1: {results[0][0][:100]}... (Score: {results[0][1]:.3f})")
        
        print(f"\n{'='*80}")
        print(f"✅ Batch completado")
        print(f"{'='*80}\n")
        
        return all_results
    
    # ------------------------------------------------------------
    # 6. Comparar con TF-IDF (opcional)
    # ------------------------------------------------------------
    def compare_with_tfidf(self, query_text, top_k=3):
        """
        Compara resultados BM25 vs TF-IDF simple.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        print(f"\n{'#'*80}")
        print(f"# COMPARACIÓN: BM25 vs TF-IDF")
        print(f"{'#'*80}\n")
        
        print(f"🔎 Consulta: '{query_text}'\n")
        
        # Resultados BM25
        print("=" * 80)
        print("BM25 (BASELINE)")
        print("=" * 80)
        bm25_results = self.query(query_text, top_k=top_k, show_details=False)
        for i, (chunk, score, _) in enumerate(bm25_results, 1):
            print(f"[{i}] Score: {score:.4f} | {chunk[:200]}...")
        
        # Resultados TF-IDF
        print("\n" + "=" * 80)
        print("TF-IDF")
        print("=" * 80)
        
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(self.chunks)
        query_vec = vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, X).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for i, idx in enumerate(top_indices, 1):
            print(f"[{i}] Score: {similarities[idx]:.4f} | {self.chunks[idx][:200]}...")
        
        print("\n" + "#" * 80 + "\n")
    
    # ------------------------------------------------------------
    # 7. Estadísticas del modelo
    # ------------------------------------------------------------
    def get_stats(self):
        """
        Retorna estadísticas del modelo baseline.
        """
        if not self.chunks:
            return {"error": "No hay chunks cargados"}
        
        stats = {
            "total_chunks": self.total_chunks,
            "metodo_chunking": "SIMPLE (sin modificaciones)",
            "metodo_retrieval": "BM25",
            "vocabulario_size": len(self.bm25.idf) if self.bm25 else 0,
            "longitud_promedio_doc": self.bm25.avgdl if self.bm25 else 0,
            "parametro_k1": self.k1,
            "parametro_b": self.b
        }
        
        print(f"\n{'='*80}")
        print(f"📊 ESTADÍSTICAS DEL MODELO BASELINE")
        print(f"{'='*80}")
        for key, value in stats.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, float):
                print(f"  • {label}: {value:.2f}")
            else:
                print(f"  • {label}: {value}")
        print(f"{'='*80}\n")
        
        return stats
    
    # ------------------------------------------------------------
    # 8. Análisis de términos más importantes
    # ------------------------------------------------------------
    def get_top_terms(self, n=20):
        """
        Retorna los términos con mayor IDF (más discriminativos).
        """
        if not self.bm25:
            raise ValueError("❌ Índice no inicializado.")
        
        # Ordenar por IDF
        sorted_terms = sorted(self.bm25.idf.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"🔝 TOP {n} TÉRMINOS MÁS DISCRIMINATIVOS (Mayor IDF)")
        print(f"{'='*80}")
        
        for i, (term, idf) in enumerate(sorted_terms[:n], 1):
            print(f"{i:2d}. {term:20s} | IDF: {idf:.4f}")
        
        print(f"{'='*80}\n")
        
        return sorted_terms[:n]


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    # Configurar rutas
    preprocessed_base = os.path.join(project_root, "data", "preprocessed")
    
    print(f"\n{'#'*80}")
    print(f"# MODELO RAG BASELINE - CHUNKING SIMPLE + BM25")
    print(f"{'#'*80}\n")
    
    # Inicializar modelo baseline
    rag_baseline = RAGBaseline(
        preprocessed_base_dir=preprocessed_base,
        k1=1.5,  # Parámetro de saturación BM25
        b=0.75   # Parámetro de normalización BM25
    )
    
    # Ejecutar pipeline completo
    rag_baseline.prepare_documents(chunk_config="processed_400_100")
    
    # Mostrar términos más discriminativos
    rag_baseline.get_top_terms(n=15)
    
    # Consultas de ejemplo
    queries = [
        "Who saves Bella from the van?",
        "Which Cullen family member is a doctor?",
        "Where does Bella move to?",
    ]
    
    print(f"\n{'#'*80}")
    print(f"# EJECUTANDO CONSULTAS DE PRUEBA")
    print(f"{'#'*80}\n")
    
    # Consultas individuales con detalles
    for query in queries:
        results = rag_baseline.query(query, top_k=3)
        input("\nPresiona Enter para continuar a la siguiente consulta...")
    
    # Mostrar estadísticas finales
    rag_baseline.get_stats()
    
    # Opcional: Comparar BM25 vs TF-IDF
    print("\n¿Deseas comparar BM25 vs TF-IDF? (s/n): ", end="")
    respuesta = input().lower()
    if respuesta == 's':
        rag_baseline.compare_with_tfidf(queries[0])
    
    # Opcional: Ejecutar batch de consultas
    print("\n¿Deseas ejecutar consultas en batch? (s/n): ", end="")
    respuesta = input().lower()
    if respuesta == 's':
        batch_results = rag_baseline.batch_query(queries, top_k=5)
        print(f"\n✅ Procesadas {len(batch_results)} consultas en batch")

# Inicializar
baseline = RAGBaseline(k1=1.5, b=0.75)

# Pipeline completo
baseline.prepare_documents("processed_400_100")

# Consultar
results = baseline.query("Who is Bella?", top_k=5)

# Batch queries
batch_results = baseline.batch_query(queries, top_k=3)

# Estadísticas
baseline.get_stats()