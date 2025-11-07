# ============================================================
# EVALUACIÓN RAG INTERACTIVA
# ============================================================

import os, sys

# Añadir la carpeta raíz del proyecto al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from openai import AzureOpenAI
from dotenv import load_dotenv
from src.utils import load_chunks_from_folder
from src.models.baseline import RAGBaseline
from src.models.model_a import RAGModelA
from src.models.model_b import RAGModelB
from src.models.model_c import RAGModelC


load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint="https://pnl-maestria.openai.azure.com/"
)

# Paths
project_root = os.path.abspath("..")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
print("Project root:", project_root)

# ===================================
# CARGAR DATOS
# ===================================
BASE_PREPROCESSED = os.path.join(project_root, "data", "preprocessed")
folders = sorted([
    os.path.join(BASE_PREPROCESSED, f)
    for f in os.listdir(BASE_PREPROCESSED)
    if f.startswith("processed_")
])

records = []
for folder in folders:
    recs = load_chunks_from_folder(folder)
    records.extend(recs)

df = pd.DataFrame.from_records(records)
documents = df["text"].astype(str).tolist()
text_to_index = {text: idx for idx, text in enumerate(documents)}

print(f" Total chunks: {len(documents)}")

# ===================================
# ÍNDICE TF-IDF MODELO BASE
# ===================================

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(documents)
print(f"✅ Índice TF-IDF: {X.shape}")

# ===================================
# FUNCIONES DE BÚSQUEDA Y GENERACIÓN
# ===================================

@lru_cache(maxsize=128)
def search_tfidf_cached(query, k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return tuple(top_k_indices), tuple(similarities[top_k_indices])

def search_tfidf(query, k=5):
    indices, scores = search_tfidf_cached(query, k)
    return list(indices), list(scores)

def generate_answer(context, query):
    """Genera respuesta usando GPT"""
    prompt = f"""You are a knowledgeable assistant who knows the Twilight Saga.
Answer naturally, briefly, like in a chat.
Use the context below ONLY if it contains the answer.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question: {query}
Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ===================================
# INICIALIZAR MODELOS
# ===================================

print("\n Inicializando modelos...")

# Baseline (BM25)
baseline = RAGBaseline(
    preprocessed_base_dir=BASE_PREPROCESSED,
    k1=1.5,
    b=0.75
)
baseline.prepare_documents(chunk_config="processed_400_100")
print(" Baseline (BM25) listo")

# Model A (FAISS + Embeddings)
try:
    model_a = RAGModelA(
        preprocessed_base_dir=BASE_PREPROCESSED,
        use_gpu=False
    )
    model_a.prepare_documents(chunk_config="processed_400_100", index_type="flat", batch_size=32)
    print(" Model A (FAISS) listo")
    model_a_available = True
except Exception as e:
    print(f"  Model A no disponible: {e}")
    model_a_available = False

# Model B (TF-IDF + Dedup 0.75)
model_b = RAGModelB(
    preprocessed_base_dir=BASE_PREPROCESSED,
    similarity_threshold=0.75
)
model_b.prepare_documents(chunk_config="processed_400_100")
print(" Model B (Dedup 0.75) listo")

# Model C (TF-IDF + Dedup 0.85)
model_c = RAGModelC(
    preprocessed_base_dir=BASE_PREPROCESSED,
    similarity_threshold=0.85
)
model_c.prepare_documents(chunk_config="processed_400_100")
print(" Model C (Dedup 0.85) listo")

# ===================================
# FUNCIÓN PARA PROCESAR QUERY INTERACTIVA
# ===================================

def process_user_query(query, k=5, show_chunks=False):
    """
    Procesa una query del usuario usando todos los modelos disponibles
    """
    print("\n" + "="*80)
    print(f"🔍 PREGUNTA: {query}")
    print("="*80)
    
    results = {}
    
    # 1. MODELO BASE (TF-IDF)
    print("\n📊 Modelo Base (TF-IDF)...")
    retrieved_indices, scores = search_tfidf(query, k=k)
    context = "\n".join([documents[i] for i in retrieved_indices])
    answer = generate_answer(context, query)
    results['base'] = {
        'answer': answer,
        'retrieved_indices': retrieved_indices,
        'scores': scores
    }
    print(f"   Respuesta: {answer}")
    
    # 2. BASELINE (BM25)
    print("\n📊 Baseline (BM25)...")
    result = baseline.query(query, top_k=k, show_details=False)
    retrieved_data = result["results"]
    retrieved_texts = [chunk for chunk, _, _ in retrieved_data]
    context = "\n".join(retrieved_texts)
    answer = generate_answer(context, query)
    results['baseline'] = {
        'answer': answer,
        'retrieved_texts': retrieved_texts
    }
    print(f"   Respuesta: {answer}")
    
    # 3. MODEL A (FAISS)
    if model_a_available:
        print("\n📊 Model A (FAISS + Embeddings)...")
        retrieved = model_a.query(query, top_k=k, show_details=False)
        retrieved_texts = [r['text'] for r in retrieved]
        context = "\n".join(retrieved_texts)
        answer = generate_answer(context, query)
        results['model_a'] = {
            'answer': answer,
            'retrieved_texts': retrieved_texts
        }
        print(f"   Respuesta: {answer}")
    
    # 4. MODEL B (TF-IDF + Dedup 0.75)
    print("\n📊 Model B (TF-IDF + Dedup 0.75)...")
    retrieved = model_b.query(query, top_k=k)
    retrieved_texts = [chunk for chunk, _, _ in retrieved]
    context = "\n".join(retrieved_texts)
    answer = generate_answer(context, query)
    results['model_b'] = {
        'answer': answer,
        'retrieved_texts': retrieved_texts
    }
    print(f"   Respuesta: {answer}")
    
    # 5. MODEL C (TF-IDF + Dedup 0.85)
    print("\n📊 Model C (TF-IDF + Dedup 0.85)...")
    retrieved = model_c.query(query, top_k=k)
    retrieved_texts = [chunk for chunk, _, _ in retrieved]
    context = "\n".join(retrieved_texts)
    answer = generate_answer(context, query)
    results['model_c'] = {
        'answer': answer,
        'retrieved_texts': retrieved_texts
    }
    print(f"   Respuesta: {answer}")
    
    # Mostrar resumen
    print("\n" + "="*80)
    print(" RESUMEN DE RESPUESTAS")
    print("="*80)
    print(f"\n🔹 Base (TF-IDF):      {results['base']['answer']}")
    print(f"🔹 Baseline (BM25):    {results['baseline']['answer']}")
    if model_a_available:
        print(f"🔹 Model A (FAISS):    {results['model_a']['answer']}")
    print(f"🔹 Model B (Dedup75):  {results['model_b']['answer']}")
    print(f"🔹 Model C (Dedup85):  {results['model_c']['answer']}")
    
    # Mostrar chunks recuperados si se solicita
    if show_chunks:
        print("\n" + "="*80)
        print("CHUNKS RECUPERADOS (Primeros 3)")
        print("="*80)
        for i, idx in enumerate(retrieved_indices[:3], 1):
            print(f"\n[Chunk {i}] (Score: {scores[i-1]:.4f})")
            print(f"{documents[idx][:300]}...")
    
    return results

# ===================================
# MODO INTERACTIVO
# ===================================

def interactive_mode():
    """
    Modo interactivo para que el usuario ingrese preguntas
    """
    print("\n" + "="*80)
    print("MODO INTERACTIVO RAG")
    print("="*80)
    print("\nIngresa tus preguntas sobre Twilight Saga.")
    print("Comandos especiales:")
    print("  • 'salir' o 'exit' - Terminar el programa")
    print("  • 'chunks' - Mostrar chunks recuperados en la próxima consulta")
    print("  • 'help' - Mostrar esta ayuda")
    print("="*80)
    
    show_chunks = False
    
    while True:
        print("\n")
        user_input = input("💬 Tu pregunta: ").strip()
        
        if not user_input:
            continue
        
        # Comandos especiales
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!")
            break
        
        if user_input.lower() == 'help':
            print("\n📖 Comandos disponibles:")
            print("  • 'salir' o 'exit' - Terminar")
            print("  • 'chunks' - Toggle mostrar chunks")
            print("  • 'help' - Esta ayuda")
            continue
        
        if user_input.lower() == 'chunks':
            show_chunks = not show_chunks
            status = "activado" if show_chunks else "desactivado ❌"
            print(f"\n📄 Mostrar chunks: {status}")
            continue
        
        # Procesar la pregunta
        try:
            process_user_query(user_input, k=5, show_chunks=show_chunks)
        except KeyboardInterrupt:
            print("\n\n  Operación cancelada")
            break
        except Exception as e:
            print(f"\n Error procesando la pregunta: {e}")
            import traceback
            traceback.print_exc()

# ===================================
# PUNTO DE ENTRADA
# ===================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" SISTEMA RAG - TWILIGHT SAGA")
    print("="*80)
    print(f"\n📚 Corpus: {len(documents)} chunks")
    print(f" Modelos disponibles: {4 if model_a_available else 3}")
    
    # Puedes elegir entre modo interactivo o evaluar queries predefinidas
    print("\n¿Qué deseas hacer?")
    print("1. Modo interactivo (ingresar preguntas)")
    print("2. Evaluación con queries predefinidas")
    
    choice = input("\nElige una opción (1 o 2): ").strip()
    
    if choice == "1":
        interactive_mode()
    else:
        # Código original de evaluación
        queries = [
            "Who saves Bella from the van?",
            "Which Cullen family member is a doctor?",
        ]
        
        print("\n Ejecutando evaluación con queries predefinidas...")
        for query in queries:
            process_user_query(query, k=5, show_chunks=True)
            time.sleep(1)  # Pausa entre queries