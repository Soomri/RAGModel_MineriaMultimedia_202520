import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import warnings

# Configurar path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class BaseRetriever(ABC):
    """
    Clase abstracta base para todos los retrievers.
    
    Define la interfaz común que deben implementar todos los métodos de recuperación.
    """
    
    def __init__(self, chunks: List[str], metadata: List[Dict] = None):
        """
        Inicializa el retriever.
        
        Parámetros:
        - chunks: Lista de textos (documentos)
        - metadata: Lista opcional de diccionarios con metadata por chunk
        """
        self.chunks = chunks
        self.metadata = metadata if metadata else [{}] * len(chunks)
        self.is_fitted = False
    
    @abstractmethod
    def fit(self):
        """
        Prepara el retriever (entrenar, indexar, etc.).
        Debe ser implementado por cada retriever específico.
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Recupera documentos relevantes para una consulta.
        
        Parámetros:
        - query: Texto de consulta
        - top_k: Número de documentos a recuperar
        
        Retorna:
        - Lista de tuplas (índice_documento, score)
        """
        pass
    
    def get_documents(self, indices: List[int]) -> List[Tuple[str, Dict]]:
        """
        Obtiene documentos completos dado sus índices.
        
        Parámetros:
        - indices: Lista de índices de documentos
        
        Retorna:
        - Lista de tuplas (texto, metadata)
        """
        return [(self.chunks[idx], self.metadata[idx]) for idx in indices]
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Método de alto nivel para consultas (incluye texto y metadata).
        
        Parámetros:
        - query: Texto de consulta
        - top_k: Número de resultados
        
        Retorna:
        - Lista de diccionarios con: {rank, index, score, text, metadata}
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever no entrenado. Ejecuta fit() primero.")
        
        # Recuperar índices y scores
        results = self.retrieve(query, top_k)
        
        # Construir respuesta completa
        formatted_results = []
        for rank, (idx, score) in enumerate(results, 1):
            formatted_results.append({
                'rank': rank,
                'index': idx,
                'score': score,
                'text': self.chunks[idx],
                'metadata': self.metadata[idx]
            })
        
        return formatted_results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Recupera documentos para múltiples consultas.
        
        Parámetros:
        - queries: Lista de consultas
        - top_k: Número de resultados por consulta
        
        Retorna:
        - Lista de listas con resultados por consulta
        """
        return [self.query(q, top_k) for q in queries]


# ============================================================
# RETRIEVERS ESPECÍFICOS
# ============================================================

class BM25Retriever(BaseRetriever):
    """
    Retriever basado en BM25 (Best Matching 25).
    Implementación del algoritmo estándar de recuperación.
    """
    
    def __init__(self, chunks: List[str], metadata: List[Dict] = None, k1: float = 1.5, b: float = 0.75):
        """
        Inicializa BM25 Retriever.
        
        Parámetros:
        - chunks: Lista de documentos
        - metadata: Metadata opcional
        - k1: Parámetro de saturación (default: 1.5)
        - b: Parámetro de normalización (default: 0.75)
        """
        super().__init__(chunks, metadata)
        self.k1 = k1
        self.b = b
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.tokenized_corpus = []
    
    def fit(self):
        """Prepara el índice BM25."""
        import math
        
        # Tokenizar documentos
        self.tokenized_corpus = [doc.lower().split() for doc in self.chunks]
        
        # Calcular longitud promedio
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.chunks)
        
        # Calcular frecuencias de documentos
        df = {}
        for document in self.tokenized_corpus:
            frequencies = set(document)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1
        
        # Calcular IDF
        corpus_size = len(self.chunks)
        for word, freq in df.items():
            self.idf[word] = math.log((corpus_size - freq + 0.5) / (freq + 0.5) + 1)
        
        self.is_fitted = True
        return self
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recupera documentos usando BM25."""
        query_terms = query.lower().split()
        scores = np.zeros(len(self.chunks))
        
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
        
        # Obtener top-k
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_k_indices]


class TFIDFRetriever(BaseRetriever):
    """
    Retriever basado en TF-IDF con similitud coseno.
    """
    
    def __init__(self, chunks: List[str], metadata: List[Dict] = None, max_features: int = 5000):
        """
        Inicializa TF-IDF Retriever.
        
        Parámetros:
        - chunks: Lista de documentos
        - metadata: Metadata opcional
        - max_features: Número máximo de features TF-IDF
        """
        super().__init__(chunks, metadata)
        self.max_features = max_features
        self.vectorizer = None
        self.X = None
    
    def fit(self):
        """Prepara el índice TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=self.max_features,
            min_df=1,
            max_df=0.95
        )
        
        self.X = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        return self
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recupera documentos usando TF-IDF."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.X).flatten()
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


class DenseRetriever(BaseRetriever):
    """
    Retriever basado en embeddings densos (Sentence Transformers).
    Requiere: pip install sentence-transformers
    """
    
    def __init__(self, chunks: List[str], metadata: List[Dict] = None, 
                 model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        """
        Inicializa Dense Retriever.
        
        Parámetros:
        - chunks: Lista de documentos
        - metadata: Metadata opcional
        - model_name: Nombre del modelo de embeddings
        """
        super().__init__(chunks, metadata)
        self.model_name = model_name
        self.model = None
        self.embeddings = None
    
    def fit(self):
        """Prepara embeddings densos."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers no instalado. Ejecuta: pip install sentence-transformers")
        
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalizar embeddings para usar producto interno como similitud coseno
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-9)
        
        self.is_fitted = True
        return self
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recupera documentos usando embeddings densos."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        
        # Similitud coseno (producto interno con vectores normalizados)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


class HybridRetriever(BaseRetriever):
    """
    Retriever híbrido que combina múltiples métodos.
    Fusiona scores de diferentes retrievers.
    """
    
    def __init__(self, chunks: List[str], metadata: List[Dict] = None, 
                 retrievers: List[BaseRetriever] = None, weights: List[float] = None):
        """
        Inicializa Hybrid Retriever.
        
        Parámetros:
        - chunks: Lista de documentos
        - metadata: Metadata opcional
        - retrievers: Lista de retrievers a combinar
        - weights: Pesos para cada retriever (deben sumar 1.0)
        """
        super().__init__(chunks, metadata)
        self.retrievers = retrievers if retrievers else []
        
        if weights:
            if len(weights) != len(self.retrievers):
                raise ValueError("Número de pesos debe coincidir con número de retrievers")
            if abs(sum(weights) - 1.0) > 1e-6:
                warnings.warn("Los pesos no suman 1.0, normalizando...")
                total = sum(weights)
                weights = [w/total for w in weights]
        else:
            # Pesos iguales por defecto
            weights = [1.0/len(self.retrievers)] * len(self.retrievers)
        
        self.weights = weights
    
    def fit(self):
        """Entrena todos los retrievers."""
        for retriever in self.retrievers:
            if not retriever.is_fitted:
                retriever.fit()
        
        self.is_fitted = True
        return self
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recupera documentos combinando múltiples retrievers."""
        # Obtener scores de cada retriever
        all_scores = np.zeros(len(self.chunks))
        
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, top_k=len(self.chunks))
            
            # Normalizar scores a [0, 1]
            scores = np.array([score for _, score in results])
            if scores.max() > 0:
                scores = scores / scores.max()
            
            # Crear array de scores en posiciones correctas
            retriever_scores = np.zeros(len(self.chunks))
            for idx, score in results:
                retriever_scores[idx] = score
            
            # Agregar scores ponderados
            all_scores += weight * retriever_scores
        
        # Obtener top-k del score combinado
        top_k_indices = np.argsort(all_scores)[::-1][:top_k]
        return [(int(idx), float(all_scores[idx])) for idx in top_k_indices]


# ============================================================
# FACTORY PARA CREAR RETRIEVERS
# ============================================================

class RetrieverFactory:
    """
    Factory para crear retrievers de forma sencilla.
    """
    
    @staticmethod
    def create(retriever_type: str, chunks: List[str], metadata: List[Dict] = None, **kwargs):
        """
        Crea un retriever del tipo especificado.
        
        Parámetros:
        - retriever_type: Tipo de retriever ('bm25', 'tfidf', 'dense', 'hybrid')
        - chunks: Lista de documentos
        - metadata: Metadata opcional
        - **kwargs: Parámetros adicionales específicos del retriever
        
        Retorna:
        - Instancia de retriever (sin entrenar, ejecutar .fit() después)
        """
        retriever_type = retriever_type.lower()
        
        if retriever_type == 'bm25':
            return BM25Retriever(chunks, metadata, **kwargs)
        
        elif retriever_type == 'tfidf':
            return TFIDFRetriever(chunks, metadata, **kwargs)
        
        elif retriever_type == 'dense':
            return DenseRetriever(chunks, metadata, **kwargs)
        
        elif retriever_type == 'hybrid':
            return HybridRetriever(chunks, metadata, **kwargs)
        
        else:
            raise ValueError(f"Tipo de retriever desconocido: {retriever_type}")
    
    @staticmethod
    def create_and_fit(retriever_type: str, chunks: List[str], metadata: List[Dict] = None, **kwargs):
        """
        Crea y entrena un retriever en un solo paso.
        
        Parámetros:
        - retriever_type: Tipo de retriever
        - chunks: Lista de documentos
        - metadata: Metadata opcional
        - **kwargs: Parámetros adicionales
        
        Retorna:
        - Instancia de retriever entrenado
        """
        retriever = RetrieverFactory.create(retriever_type, chunks, metadata, **kwargs)
        retriever.fit()
        return retriever


# ============================================================
# UTILIDADES
# ============================================================

def compare_retrievers(query: str, retrievers: Dict[str, BaseRetriever], top_k: int = 5):
    """
    Compara resultados de múltiples retrievers para una consulta.
    
    Parámetros:
    - query: Texto de consulta
    - retrievers: Diccionario {nombre: retriever}
    - top_k: Número de resultados
    """
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN DE RETRIEVERS")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"Top-K: {top_k}\n")
    
    for name, retriever in retrievers.items():
        print(f"\n{'-'*80}")
        print(f" {name.upper()}")
        print(f"{'-'*80}")
        
        results = retriever.query(query, top_k)
        
        for r in results:
            print(f"[{r['rank']}] Score: {r['score']:.4f} | {r['text'][:150]}...")
    
    print(f"\n{'='*80}\n")


def evaluate_retriever(retriever: BaseRetriever, queries: List[str], 
                       ground_truth: List[List[int]], k: int = 5):
    """
    Evalúa un retriever con métricas de recall y precision.
    
    Parámetros:
    - retriever: Instancia de retriever
    - queries: Lista de consultas
    - ground_truth: Lista de listas con índices relevantes por consulta
    - k: Número de resultados a evaluar
    
    Retorna:
    - Diccionario con métricas
    """
    recalls = []
    precisions = []
    
    for query, relevant in zip(queries, ground_truth):
        results = retriever.retrieve(query, top_k=k)
        retrieved = [idx for idx, _ in results]
        
        hits = len(set(retrieved) & set(relevant))
        
        recall = hits / len(relevant) if relevant else 0
        precision = hits / k
        
        recalls.append(recall)
        precisions.append(precision)
    
    metrics = {
        'recall@k': np.mean(recalls),
        'precision@k': np.mean(precisions),
        'recalls': recalls,
        'precisions': precisions
    }
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUACIÓN DEL RETRIEVER")
    print(f"{'='*80}")
    print(f"Recall@{k}:    {metrics['recall@k']:.3f}")
    print(f"Precision@{k}: {metrics['precision@k']:.3f}")
    print(f"{'='*80}\n")
    
    return metrics


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print(f"# RETRIEVER - SISTEMA DE RECUPERACIÓN UNIFICADO")
    print(f"{'#'*80}\n")
    
    # Documentos de ejemplo (simulados)
    chunks = [
        "Bella Swan moves to Forks, Washington to live with her father Charlie.",
        "Edward Cullen is a vampire who falls in love with Bella.",
        "The Cullen family feeds on animal blood instead of human blood.",
        "Carlisle Cullen is the patriarch and works as a doctor.",
        "Jacob Black is a werewolf and Bella's childhood friend.",
        "Edward saves Bella from a van accident using his vampire strength."
    ]
    
    metadata = [
        {'book': 'twilight', 'chapter': 1},
        {'book': 'twilight', 'chapter': 2},
        {'book': 'twilight', 'chapter': 3},
        {'book': 'twilight', 'chapter': 4},
        {'book': 'newmoon', 'chapter': 1},
        {'book': 'twilight', 'chapter': 5}
    ]
    
    # Crear y entrenar diferentes retrievers
    print("Creando retrievers...\n")
    
    bm25 = RetrieverFactory.create_and_fit('bm25', chunks, metadata, k1=1.5, b=0.75)
    print("BM25 Retriever creado")
    
    tfidf = RetrieverFactory.create_and_fit('tfidf', chunks, metadata, max_features=1000)
    print("TF-IDF Retriever creado")
    
    # Dense requiere sentence-transformers (opcional)
    try:
        dense = RetrieverFactory.create_and_fit('dense', chunks, metadata)
        print("Dense Retriever creado")
        has_dense = True
    except ImportError:
        print("Dense Retriever no disponible (falta sentence-transformers)")
        has_dense = False
    
    # Crear retriever híbrido
    if has_dense:
        hybrid = HybridRetriever(
            chunks, 
            metadata,
            retrievers=[bm25, tfidf, dense],
            weights=[0.4, 0.3, 0.3]
        )
    else:
        hybrid = HybridRetriever(
            chunks,
            metadata,
            retrievers=[bm25, tfidf],
            weights=[0.5, 0.5]
        )
    
    hybrid.fit()
    print("Hybrid Retriever creado\n")
    
    # Consulta de ejemplo
    query = "Who saves Bella from danger?"
    
    # Comparar retrievers
    retrievers_dict = {
        'BM25': bm25,
        'TF-IDF': tfidf,
        'Hybrid': hybrid
    }
    
    if has_dense:
        retrievers_dict['Dense'] = dense
    
    compare_retrievers(query, retrievers_dict, top_k=3)
    
    # Evaluación con ground truth simulado
    test_queries = [
        "Who is Bella's father?",
        "Who saves Bella from the van?"
    ]
    
    ground_truth = [
        [0],  # Query 1: chunk 0 es relevante
        [5]   # Query 2: chunk 5 es relevante
    ]
    
    print("\n Evaluando BM25...")
    evaluate_retriever(bm25, test_queries, ground_truth, k=3)
    
    print("\n Evaluando TF-IDF...")
    evaluate_retriever(tfidf, test_queries, ground_truth, k=3)
    
    print("\n Demo completada")