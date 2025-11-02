import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurar path para importar módulos
# Desde src/models/model_c.py necesitamos subir 2 niveles
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")

# Agregar al path si no está
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Ahora importar directamente desde utils (no src.utils)
from utils import load_chunks_from_folder


class RAGModelC:
    """
    Modelo RAG con chunking optimizado + deduplicación (Model C).
    
    Pipeline:
    1. Cargar chunks preprocesados (del notebook 02)
    2. Aplicar deduplicación por similitud
    3. Crear índice TF-IDF
    4. Realizar consultas
    """

    def __init__(self, preprocessed_base_dir=None, similarity_threshold=0.85):
        """
        Inicializa el modelo RAG C.
        
        Parámetros:
        - preprocessed_base_dir: Ruta base a data/preprocessed
        - similarity_threshold: Umbral de similitud para deduplicación (0-1)
        """
        if preprocessed_base_dir is None:
            self.preprocessed_base_dir = os.path.join(project_root, "data", "preprocessed")
        else:
            self.preprocessed_base_dir = preprocessed_base_dir
            
        self.similarity_threshold = similarity_threshold
        self.chunks = []
        self.chunks_metadata = []
        self.vectorizer = None
        self.X = None
        self.original_count = 0
        self.deduplicated_count = 0

    # ------------------------------------------------------------
    # 1. Cargar chunks preprocesados
    # ------------------------------------------------------------
    def load_preprocessed_chunks(self, chunk_config="processed_400_100"):
        """
        Carga chunks desde las carpetas preprocesadas.
        
        Parámetros:
        - chunk_config: Nombre de la carpeta (ej: "processed_400_100")
        """
        print(f"\n{'='*80}")
        print(f"📂 CARGANDO CHUNKS PREPROCESADOS")
        print(f"{'='*80}")
        
        folder_path = os.path.join(self.preprocessed_base_dir, chunk_config)
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No se encuentra la carpeta: {folder_path}")
        
        print(f"Carpeta: {folder_path}")
        
        # Cargar usando la función de utils
        records = load_chunks_from_folder(folder_path)
        
        if not records:
            raise ValueError(f"No se encontraron chunks en {folder_path}")
        
        df = pd.DataFrame.from_records(records)
        
        # Extraer textos y metadata
        self.chunks = df['text'].astype(str).tolist()
        self.chunks_metadata = df.to_dict('records')
        self.original_count = len(self.chunks)
        
        print(f"Cargados {self.original_count} chunks")
        print(f"Configuración: {chunk_config}")
        print(f"{'='*80}\n")
        
        return self.chunks

    # ------------------------------------------------------------
    # 2. Deduplicación por similitud
    # ------------------------------------------------------------
    def deduplicate_chunks(self):
        """
        Elimina chunks redundantes usando similitud de coseno.
        Mantiene el chunk más representativo de cada grupo similar.
        """
        if not self.chunks:
            raise ValueError("No hay chunks cargados. Ejecuta load_preprocessed_chunks() primero.")
        
        print(f"\n{'='*80}")
        print(f"DEDUPLICACIÓN DE CHUNKS")
        print(f"{'='*80}")
        print(f"Chunks originales: {len(self.chunks)}")
        print(f"Umbral de similitud: {self.similarity_threshold}")
        
        # Crear representación TF-IDF temporal para comparación
        temp_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000,
            min_df=1,
            max_df=0.95
        )
        
        try:
            temp_X = temp_vectorizer.fit_transform(self.chunks)
        except ValueError as e:
            print(f"Error en vectorización: {e}")
            print("Continuando sin deduplicación...")
            self.deduplicated_count = len(self.chunks)
            return self.chunks
        
        # Calcular matriz de similitud
        print("Calculando similitudes...")
        similarity_matrix = cosine_similarity(temp_X)
        
        # Identificar chunks únicos
        unique_chunks = []
        unique_metadata = []
        seen = set()
        duplicates_count = 0
        
        for i in range(len(self.chunks)):
            if i in seen:
                continue
            
            # Encontrar todos los chunks similares
            similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
            
            # Marcar como vistos
            seen.update(similar_indices)
            
            # Mantener el chunk con más palabras (más informativo)
            best_idx = max(similar_indices, key=lambda idx: len(self.chunks[idx].split()))
            unique_chunks.append(self.chunks[best_idx])
            unique_metadata.append(self.chunks_metadata[best_idx])
            
            if len(similar_indices) > 1:
                duplicates_count += len(similar_indices) - 1
        
        # Actualizar chunks
        self.chunks = unique_chunks
        self.chunks_metadata = unique_metadata
        self.deduplicated_count = len(self.chunks)
        
        reduction_pct = (1 - self.deduplicated_count / self.original_count) * 100
        
        print(f"Deduplicación completada")
        print(f"Chunks eliminados: {duplicates_count}")
        print(f"Chunks únicos: {self.deduplicated_count}/{self.original_count}")
        print(f"Reducción: {reduction_pct:.1f}%")
        print(f"{'='*80}\n")
        
        return self.chunks

    # ------------------------------------------------------------
    # 3. Crear índice TF-IDF
    # ------------------------------------------------------------
    def create_index(self):
        """
        Crea el índice TF-IDF para los chunks deduplicados.
        """
        if not self.chunks:
            raise ValueError("No hay chunks para indexar.")
        
        print(f"\n{'='*80}")
        print(f"CREACIÓN DE ÍNDICE TF-IDF")
        print(f"{'='*80}")
        
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        self.X = self.vectorizer.fit_transform(self.chunks)
        
        print(f"Índice creado")
        print(f"Documentos: {self.X.shape[0]}")
        print(f"Features: {self.X.shape[1]}")
        print(f"{'='*80}\n")
        
        return self.vectorizer, self.X

    # ------------------------------------------------------------
    # 4. Pipeline completo: cargar + deduplicar + indexar
    # ------------------------------------------------------------
    def prepare_documents(self, chunk_config="processed_400_100"):
        """
        Pipeline completo: carga, deduplica e indexa chunks.
        
        Parámetros:
        - chunk_config: Configuración de chunks a usar
        """
        print(f"\n{'#'*80}")
        print(f"# PIPELINE MODEL C: CHUNKING OPTIMIZADO + DEDUPLICACIÓN")
        print(f"{'#'*80}\n")
        
        # Paso 1: Cargar chunks preprocesados
        self.load_preprocessed_chunks(chunk_config)
        
        # Paso 2: Deduplicación
        self.deduplicate_chunks()
        
        # Paso 3: Indexación
        self.create_index()
        
        print(f"\n{'#'*80}")
        print(f"# PIPELINE COMPLETADO")
        print(f"{'#'*80}")
        print(f"Resumen:")
        print(f"  • Chunks originales:   {self.original_count}")
        print(f"  • Chunks deduplicados: {self.deduplicated_count}")
        print(f"  • Reducción:           {(1 - self.deduplicated_count/self.original_count)*100:.1f}%")
        print(f"  • Features TF-IDF:     {self.X.shape[1]}")
        print(f"{'#'*80}\n")

    # ------------------------------------------------------------
    # 5. Consulta
    # ------------------------------------------------------------
    def query(self, query_text, top_k=3, show_details=True):
        """
        Realiza una consulta al sistema RAG.
        
        Parámetros:
        - query_text: Texto de la consulta
        - top_k: Número de resultados a retornar
        - show_details: Mostrar detalles de los resultados
        
        Retorna:
        - Lista de tuplas (chunk, score, metadata)
        """
        if self.vectorizer is None or self.X is None:
            raise ValueError("Índice no inicializado. Ejecuta prepare_documents() primero.")
        
        if show_details:
            print(f"\n{'='*80}")
            print(f"CONSULTA: '{query_text}'")
            print(f"{'='*80}\n")
        
        # Vectorizar consulta
        query_vec = self.vectorizer.transform([query_text])
        
        # Calcular similitudes
        similarities = cosine_similarity(query_vec, self.X).flatten()
        
        # Obtener top-k índices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Preparar resultados
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            chunk = self.chunks[idx]
            score = similarities[idx]
            metadata = self.chunks_metadata[idx]
            
            results.append((chunk, score, metadata))
            
            if show_details:
                print(f"RANK {rank} | Score: {score:.4f}")
                print(f"Libro: {metadata.get('book_name', 'N/A')}")
                print(f"Chunk #{metadata.get('chunk_number', 'N/A')}")
                print(f"Palabras: {metadata.get('word_count', 'N/A')}")
                print(f"Texto: {chunk[:300]}...")
                print("-" * 80)
        
        #if show_details:
          #  print(f"\n RESPUESTA BASADA EN CONTEXTO MÁS RELEVANTE:")
           # print(f"{results[0][0][:500]}...")
           # print("=" * 80 + "\n")
        
        return results

    # ------------------------------------------------------------
    # 6. Comparar con/sin deduplicación
    # ------------------------------------------------------------
    def compare_with_without_dedup(self, query_text, top_k=3):
        """
        Compara resultados con y sin deduplicación.
        """
        print(f"\n{'#'*80}")
        print(f"# COMPARACIÓN: CON vs SIN DEDUPLICACIÓN")
        print(f"{'#'*80}\n")
        
        # Guardar estado actual
        chunks_dedup = self.chunks.copy()
        X_dedup = self.X
        vectorizer_dedup = self.vectorizer
        
        # Crear índice sin deduplicación
        print("Recreando índice SIN deduplicación...")
        
        # Recargar chunks originales
        temp_records = load_chunks_from_folder(
            os.path.join(self.preprocessed_base_dir, "processed_400_100")
        )
        chunks_original = [r['text'] for r in temp_records]
        
        vectorizer_no_dedup = TfidfVectorizer(stop_words="english", max_features=5000)
        X_no_dedup = vectorizer_no_dedup.fit_transform(chunks_original)
        
        print(f"Índices creados")
        print(f"  • Sin dedup: {X_no_dedup.shape[0]} chunks")
        print(f"  • Con dedup: {X_dedup.shape[0]} chunks\n")
        
        # Consultar ambos
        print(f"Consulta: '{query_text}'\n")
        
        print("=" * 80)
        print("SIN DEDUPLICACIÓN")
        print("=" * 80)
        query_vec = vectorizer_no_dedup.transform([query_text])
        sims = cosine_similarity(query_vec, X_no_dedup).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        for i, idx in enumerate(top_idx, 1):
            print(f"[{i}] Score: {sims[idx]:.4f} | {chunks_original[idx][:200]}...")
        
        print("\n" + "=" * 80)
        print("CON DEDUPLICACIÓN")
        print("=" * 80)
        results = self.query(query_text, top_k, show_details=False)
        for i, (chunk, score, _) in enumerate(results, 1):
            print(f"[{i}] Score: {score:.4f} | {chunk[:200]}...")
        
        print("\n" + "#" * 80 + "\n")

    # ------------------------------------------------------------
    # 7. Estadísticas del modelo
    # ------------------------------------------------------------
    def get_stats(self):
        """
        Retorna estadísticas del modelo.
        """
        if not self.chunks:
            return {"error": "No hay chunks cargados"}
        
        stats = {
            "chunks_originales": self.original_count,
            "chunks_deduplicados": self.deduplicated_count,
            "reduccion_porcentaje": (1 - self.deduplicated_count/self.original_count)*100 if self.original_count > 0 else 0,
            "features_tfidf": self.X.shape[1] if self.X is not None else 0,
            "umbral_similitud": self.similarity_threshold,
            "vocabulario_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }
        
        print(f"\n{'='*80}")
        print(f"ESTADÍSTICAS DEL MODELO C")
        print(f"{'='*80}")
        for key, value in stats.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        print(f"{'='*80}\n")
        
        return stats


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    # Configurar rutas (ya tenemos project_root definido arriba)
    preprocessed_base = os.path.join(project_root, "data", "preprocessed")
    
    print(f"\n{'#'*80}")
    print(f"# MODELO RAG C - CHUNKING OPTIMIZADO + DEDUPLICACIÓN")
    print(f"{'#'*80}\n")
    
    # Inicializar modelo
    rag_c = RAGModelC(
        preprocessed_base_dir=preprocessed_base,
        similarity_threshold=0.85
    )
    
    # Ejecutar pipeline completo
    rag_c.prepare_documents(chunk_config="processed_400_100")
    
    # Consultas de ejemplo
    queries = [
        "Who saves Bella from the van?",
        "Which Cullen family member is a doctor?",
        "Where does Bella move to?",
    ]
    
    print(f"\n{'#'*80}")
    print(f"# EJECUTANDO CONSULTAS DE PRUEBA")
    print(f"{'#'*80}\n")
    
    for query in queries:
        results = rag_c.query(query, top_k=3)
        input("\nPresiona Enter para continuar a la siguiente consulta...")
    
    # Mostrar estadísticas finales
    rag_c.get_stats()
    
    # Opcional: Comparar con/sin deduplicación
    print("\n¿Deseas comparar resultados con/sin deduplicación? (s/n): ", end="")
    respuesta = input().lower()
    if respuesta == 's':
        rag_c.compare_with_without_dedup(queries[0])