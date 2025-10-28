import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

# Configurar path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils import load_chunks_from_folder


class RAGModelA:
    """
    Modelo RAG A: Embeddings semánticos densos + FAISS.
    
    Pipeline:
    1. Cargar chunks preprocesados
    2. Generar embeddings densos con Sentence Transformers
    3. Indexar con FAISS (búsqueda vectorial eficiente)
    4. Realizar consultas semánticas
    
    Este modelo usa búsqueda semántica avanzada, capturando mejor
    el significado de las consultas vs métodos léxicos como BM25.
    """
    
    def __init__(self, preprocessed_base_dir=None, 
                 model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                 use_gpu=False):
        """
        Inicializa el modelo RAG A.
        
        Parámetros:
        - preprocessed_base_dir: Ruta base a data/preprocessed
        - model_name: Nombre del modelo de embeddings
        - use_gpu: Usar GPU si está disponible
        """
        if preprocessed_base_dir is None:
            self.preprocessed_base_dir = os.path.join(project_root, "data", "preprocessed")
        else:
            self.preprocessed_base_dir = preprocessed_base_dir
        
        self.model_name = model_name
        self.use_gpu = use_gpu
        
        self.chunks = []
        self.chunks_metadata = []
        self.embeddings = None
        self.embedding_model = None
        self.faiss_index = None
        self.total_chunks = 0
        self.embedding_dim = None
    
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
        print(f"CARGANDO CHUNKS PREPROCESADOS")
        print(f"{'='*80}")
        
        folder_path = os.path.join(self.preprocessed_base_dir, chunk_config)
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ No se encuentra la carpeta: {folder_path}")
        
        print(f"📁 Carpeta: {folder_path}")
        
        records = load_chunks_from_folder(folder_path)
        
        if not records:
            raise ValueError(f"No se encontraron chunks en {folder_path}")
        
        df = pd.DataFrame.from_records(records)
        
        self.chunks = df['text'].astype(str).tolist()
        self.chunks_metadata = df.to_dict('records')
        self.total_chunks = len(self.chunks)
        
        print(f"Cargados {self.total_chunks} chunks")
        print(f"Configuración: {chunk_config}")
        print(f"{'='*80}\n")
        
        return self.chunks
    
    # ------------------------------------------------------------
    # 2. Cargar modelo de embeddings
    # ------------------------------------------------------------
    def load_embedding_model(self):
        """
        Carga el modelo de Sentence Transformers para generar embeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers no instalado.\n"
                "Instalar con: pip install sentence-transformers"
            )
        
        print(f"\n{'='*80}")
        print(f"CARGANDO MODELO DE EMBEDDINGS")
        print(f"{'='*80}")
        print(f"Modelo: {self.model_name}")
        
        device = 'cuda' if self.use_gpu else 'cpu'
        
        try:
            self.embedding_model = SentenceTransformer(self.model_name, device=device)
            print(f"Modelo cargado en: {device}")
            
            # Obtener dimensión de embeddings
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]
            print(f"Dimensión embeddings: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            print(f"Intentando con modelo alternativo...")
            self.model_name = "all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(self.model_name, device=device)
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]
            print(f"Modelo alternativo cargado: {self.model_name}")
        
        print(f"{'='*80}\n")
        
        return self.embedding_model
    
    # ------------------------------------------------------------
    # 3. Generar embeddings densos
    # ------------------------------------------------------------
    def generate_embeddings(self, batch_size=32):
        """
        Genera embeddings densos para todos los chunks.
        
        Parámetros:
        - batch_size: Tamaño de batch para procesamiento
        """
        if not self.chunks:
            raise ValueError("No hay chunks cargados.")
        
        if self.embedding_model is None:
            self.load_embedding_model()
        
        print(f"\n{'='*80}")
        print(f"GENERANDO EMBEDDINGS DENSOS")
        print(f"{'='*80}")
        print(f"Total chunks: {len(self.chunks)}")
        print(f"Batch size: {batch_size}")
        
        # Generar embeddings
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar para similitud coseno
        )
        
        print(f"Embeddings generados")
        print(f"Shape: {self.embeddings.shape}")
        print(f"Tamaño en memoria: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
        print(f"{'='*80}\n")
        
        return self.embeddings
    
    # ------------------------------------------------------------
    # 4. Crear índice FAISS
    # ------------------------------------------------------------
    def create_faiss_index(self, index_type="flat"):
        """
        Crea índice FAISS para búsqueda vectorial eficiente.
        
        Parámetros:
        - index_type: Tipo de índice ('flat', 'ivf', 'hnsw')
            - 'flat': Búsqueda exacta (más lento, más preciso)
            - 'ivf': Inverted File Index (rápido, aproximado)
            - 'hnsw': Hierarchical NSW (muy rápido, aproximado)
        """
        if self.embeddings is None:
            raise ValueError("Embeddings no generados. Ejecuta generate_embeddings() primero.")
        
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss no instalado.\n"
                "Instalar con: pip install faiss-cpu  (o faiss-gpu para GPU)"
            )
        
        print(f"\n{'='*80}")
        print(f"CREANDO ÍNDICE FAISS")
        print(f"{'='*80}")
        print(f"Tipo de índice: {index_type.upper()}")
        
        d = self.embedding_dim
        
        if index_type == "flat":
            # IndexFlatIP: Producto interno (equivale a coseno con vectores normalizados)
            self.faiss_index = faiss.IndexFlatIP(d)
            print(f"Usando IndexFlatIP (búsqueda exacta)")
        
        elif index_type == "ivf":
            # IVF: Más rápido pero aproximado
            nlist = min(100, len(self.chunks) // 10)  # Número de clusters
            quantizer = faiss.IndexFlatIP(d)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            
            print(f"Usando IndexIVFFlat (búsqueda aproximada)")
            print(f"Clusters: {nlist}")
            print(f"Entrenando índice...")
            
            # Entrenar IVF
            self.faiss_index.train(self.embeddings.astype('float32'))
            print(f"Índice entrenado")
        
        elif index_type == "hnsw":
            # HNSW: Muy rápido, buena precisión
            M = 32  # Número de conexiones por nodo
            self.faiss_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            print(f"Usando IndexHNSWFlat (grafo jerárquico)")
            print(f"Conexiones por nodo: {M}")
        
        else:
            raise ValueError(f"Tipo de índice desconocido: {index_type}")
        
        # Agregar embeddings al índice
        print(f"Agregando {len(self.embeddings)} vectores al índice...")
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"Índice FAISS creado")
        print(f"Total vectores: {self.faiss_index.ntotal}")
        print(f"{'='*80}\n")
        
        return self.faiss_index
    
    # ------------------------------------------------------------
    # 5. Pipeline completo: cargar + embedd + indexar
    # ------------------------------------------------------------
    def prepare_documents(self, chunk_config="processed_400_100", 
                         index_type="flat", batch_size=32):
        """
        Pipeline completo: carga, genera embeddings e indexa.
        
        Parámetros:
        - chunk_config: Configuración de chunks
        - index_type: Tipo de índice FAISS
        - batch_size: Tamaño de batch para embeddings
        """
        print(f"\n{'#'*80}")
        print(f"# PIPELINE MODEL A: EMBEDDINGS DENSOS + FAISS")
        print(f"{'#'*80}\n")
        
        # Paso 1: Cargar chunks
        self.load_preprocessed_chunks(chunk_config)
        
        # Paso 2: Cargar modelo de embeddings
        self.load_embedding_model()
        
        # Paso 3: Generar embeddings
        self.generate_embeddings(batch_size=batch_size)
        
        # Paso 4: Crear índice FAISS
        self.create_faiss_index(index_type=index_type)
        
        print(f"\n{'#'*80}")
        print(f"#PIPELINE COMPLETADO")
        print(f"{'#'*80}")
        print(f"Resumen:")
        print(f"  • Total chunks:        {self.total_chunks}")
        print(f"  • Modelo embeddings:   {self.model_name}")
        print(f"  • Dimensión:           {self.embedding_dim}")
        print(f"  • Tipo índice FAISS:   {index_type.upper()}")
        print(f"  • Vectores indexados:  {self.faiss_index.ntotal}")
        print(f"{'#'*80}\n")
    
    # ------------------------------------------------------------
    # 6. Búsqueda semántica
    # ------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Búsqueda semántica usando FAISS.
        
        Parámetros:
        - query: Texto de consulta
        - top_k: Número de resultados
        
        Retorna:
        - Lista de tuplas (índice, score)
        """
        if self.faiss_index is None:
            raise ValueError("Índice no inicializado. Ejecuta prepare_documents() primero.")
        
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Buscar en FAISS
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convertir a lista de tuplas
        results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]
        
        return results
    
    # ------------------------------------------------------------
    # 7. Consulta de alto nivel
    # ------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 3, show_details: bool = True) -> List[Dict[str, Any]]:
        """
        Realiza una consulta semántica con detalles completos.
        
        Parámetros:
        - query_text: Texto de la consulta
        - top_k: Número de resultados
        - show_details: Mostrar detalles de resultados
        
        Retorna:
        - Lista de diccionarios con resultados completos
        """
        if show_details:
            print(f"\n{'='*80}")
            print(f"CONSULTA SEMÁNTICA: '{query_text}'")
            print(f"{'='*80}\n")
        
        # Buscar
        search_results = self.search(query_text, top_k)
        
        # Formatear resultados
        results = []
        for rank, (idx, score) in enumerate(search_results, 1):
            chunk = self.chunks[idx]
            metadata = self.chunks_metadata[idx]
            
            result = {
                'rank': rank,
                'index': idx,
                'score': score,
                'text': chunk,
                'metadata': metadata
            }
            
            results.append(result)
            
            if show_details:
                print(f"RANK {rank} | Similarity: {score:.4f}")
                print(f"Libro: {metadata.get('book_name', 'N/A')}")
                print(f"Chunk #{metadata.get('chunk_number', 'N/A')}")
                print(f"Palabras: {metadata.get('word_count', 'N/A')}")
                print(f"Texto: {chunk[:300]}...")
                print("-" * 80)
        
        if show_details and results:
            print(f"\n RESPUESTA BASADA EN CONTEXTO MÁS RELEVANTE:")
            print(f"{results[0]['text'][:500]}...")
            print("=" * 80 + "\n")
        
        return results
    
    # ------------------------------------------------------------
    # 8. Búsqueda batch
    # ------------------------------------------------------------
    def batch_query(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        Ejecuta múltiples consultas de forma eficiente.
        
        Parámetros:
        - queries: Lista de consultas
        - top_k: Número de resultados por consulta
        
        Retorna:
        - Lista de listas con resultados
        """
        print(f"\n{'='*80}")
        print(f"EJECUCIÓN BATCH: {len(queries)} consultas")
        print(f"{'='*80}\n")
        
        all_results = []
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")
            results = self.query(query, top_k=top_k, show_details=False)
            all_results.append(results)
            
            if results:
                print(f"  → Top-1: {results[0]['text'][:100]}... (Score: {results[0]['score']:.3f})")
        
        print(f"\n{'='*80}")
        print(f"Batch completado")
        print(f"{'='*80}\n")
        
        return all_results
    
    # ------------------------------------------------------------
    # 9. Guardar y cargar índice
    # ------------------------------------------------------------
    def save_index(self, output_dir: str):
        """
        Guarda el índice FAISS y embeddings en disco.
        
        Parámetros:
        - output_dir: Directorio de salida
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss no disponible")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar índice FAISS
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.faiss_index, index_path)
        
        # Guardar embeddings
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        
        # Guardar metadata
        metadata_path = os.path.join(output_dir, "chunks_metadata.parquet")
        df_meta = pd.DataFrame(self.chunks_metadata)
        df_meta['chunk_text'] = self.chunks
        df_meta.to_parquet(metadata_path, index=False)
        
        # Guardar configuración
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_chunks': self.total_chunks
        }
        
        import json
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n Índice guardado en: {output_dir}")
        print(f"Archivos:")
        print(f"  • {index_path}")
        print(f"  • {embeddings_path}")
        print(f"  • {metadata_path}")
        print(f"  • {config_path}\n")
    
    def load_index(self, input_dir: str):
        """
        Carga índice FAISS y embeddings desde disco.
        
        Parámetros:
        - input_dir: Directorio con archivos guardados
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss no disponible")
        
        print(f"\n{'='*80}")
        print(f"CARGANDO ÍNDICE DESDE DISCO")
        print(f"{'='*80}")
        print(f"Directorio: {input_dir}")
        
        # Cargar configuración
        import json
        config_path = os.path.join(input_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.embedding_dim = config['embedding_dim']
        self.total_chunks = config['total_chunks']
        
        # Cargar modelo de embeddings
        self.load_embedding_model()
        
        # Cargar índice FAISS
        index_path = os.path.join(input_dir, "faiss_index.bin")
        self.faiss_index = faiss.read_index(index_path)
        
        # Cargar embeddings
        embeddings_path = os.path.join(input_dir, "embeddings.npy")
        self.embeddings = np.load(embeddings_path)
        
        # Cargar metadata
        metadata_path = os.path.join(input_dir, "chunks_metadata.parquet")
        df_meta = pd.read_parquet(metadata_path)
        self.chunks = df_meta['chunk_text'].tolist()
        self.chunks_metadata = df_meta.drop('chunk_text', axis=1).to_dict('records')
        
        print(f"Índice cargado")
        print(f"Vectores: {self.faiss_index.ntotal}")
        print(f"Chunks: {len(self.chunks)}")
        print(f"{'='*80}\n")
    
    # ------------------------------------------------------------
    # 10. Estadísticas del modelo
    # ------------------------------------------------------------
    def get_stats(self):
        """
        Retorna estadísticas del modelo.
        """
        if not self.chunks:
            return {"error": "No hay chunks cargados"}
        
        stats = {
            "total_chunks": self.total_chunks,
            "modelo_embeddings": self.model_name,
            "dimension_embeddings": self.embedding_dim,
            "vectores_indexados": self.faiss_index.ntotal if self.faiss_index else 0,
            "tamano_embeddings_mb": self.embeddings.nbytes / 1024 / 1024 if self.embeddings is not None else 0,
            "metodo_busqueda": "FAISS (búsqueda vectorial densa)"
        }
        
        print(f"\n{'='*80}")
        print(f"ESTADÍSTICAS DEL MODELO A")
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
    # 11. Visualización de embeddings (opcional)
    # ------------------------------------------------------------
    def visualize_embeddings(self, n_samples=100, method='tsne'):
        """
        Visualiza embeddings en 2D usando t-SNE o UMAP.
        
        Parámetros:
        - n_samples: Número de muestras a visualizar
        - method: 'tsne' o 'umap'
        """
        if self.embeddings is None:
            raise ValueError("Embeddings no generados")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib no instalado")
        
        # Muestrear embeddings
        indices = np.random.choice(len(self.embeddings), min(n_samples, len(self.embeddings)), replace=False)
        sample_embeddings = self.embeddings[indices]
        
        # Reducir dimensionalidad
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                raise ImportError("umap no instalado. Instalar con: pip install umap-learn")
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        print(f"Reduciendo dimensionalidad con {method.upper()}...")
        embeddings_2d = reducer.fit_transform(sample_embeddings)
        
        # Visualizar
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title(f'Visualización de Embeddings ({method.upper()})')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(project_root, 'data', f'embeddings_viz_{method}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {output_path}")
        plt.show()


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    preprocessed_base = os.path.join(project_root, "data", "preprocessed")
    
    print(f"\n{'#'*80}")
    print(f"# MODELO RAG A - EMBEDDINGS DENSOS + FAISS")
    print(f"{'#'*80}\n")
    
    # Inicializar modelo
    rag_a = RAGModelA(
        preprocessed_base_dir=preprocessed_base,
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        use_gpu=False
    )
    
    # Ejecutar pipeline completo
    rag_a.prepare_documents(
        chunk_config="processed_400_100",
        index_type="flat",  # 'flat', 'ivf', o 'hnsw'
        batch_size=32
    )
    
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
        results = rag_a.query(query, top_k=3)
        input("\nPresiona Enter para continuar a la siguiente consulta...")
    
    # Mostrar estadísticas
    rag_a.get_stats()
    
    # Guardar índice
    print("\n¿Deseas guardar el índice? (s/n): ", end="")
    respuesta = input().lower()
    if respuesta == 's':
        output_dir = os.path.join(project_root, "data", "index", "model_a")
        rag_a.save_index(output_dir)
    
    # Visualizar embeddings
    print("\n¿Deseas visualizar embeddings? (s/n): ", end="")
    respuesta = input().lower()
    if respuesta == 's':
        try:
            rag_a.visualize_embeddings(n_samples=50, method='tsne')
        except Exception as e:
            print(f"Error en visualización: {e}")

# Inicializar
rag_a = RAGModelA(use_gpu=False)

# Preparar (una sola vez)
rag_a.prepare_documents("processed_400_100", index_type="flat")

# Guardar para reutilizar
rag_a.save_index("data/index/model_a")

# Cargar en futuras ejecuciones
rag_a.load_index("data/index/model_a")

# Consultar
results = rag_a.query("Who is Bella?", top_k=5)

# Batch
batch_results = rag_a.batch_query(queries, top_k=3)