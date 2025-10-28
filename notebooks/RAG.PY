# ============================================================
# NOTEBOOK COMPLETO: Evaluación RAG con respuestas estilo chat
# ============================================================
import os, sys

# Añadir la carpeta raíz del proyecto al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


# ===================================
# 1. Setup y dependencias
# ===================================
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from openai import AzureOpenAI
from dotenv import load_dotenv
from src.utils import load_chunks_from_folder
from src.models.model_b import RAGModelB
from src.models.model_c import RAGModelC


load_dotenv()

# Verificar clave
print("🔑 Clave cargada:", bool(os.getenv("AZURE_OPENAI_API_KEY")))

# Conexión a Azure OpenAI
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
print("✅ Project root:", project_root)

# ===================================
# 2. Cargar chunks e índice TF-IDF (Modelo Base)
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
print(f"📄 Total chunks cargados: {len(documents)}")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(documents)
print(f"✅ Índice TF-IDF recreado: {X.shape[0]} documentos, {X.shape[1]} features")

# ===================================
# 3. Funciones de evaluación
# ===================================
@lru_cache(maxsize=128)
def get_similarities_cached(query, k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_scores = similarities[top_k_indices]
    return tuple(top_k_indices), tuple(top_k_scores)

def recall_at_k(top_k_indices, relevant_indices):
    if not relevant_indices:
        return 0.0
    hits = len(set(top_k_indices) & set(relevant_indices))
    return hits / len(relevant_indices)

def precision_at_k(top_k_indices, relevant_indices, k=5):
    if not relevant_indices:
        return 0.0
    hits = len(set(top_k_indices) & set(relevant_indices))
    return hits / k

def average_context_size(retrieved_indices, documents):
    chunks = [documents[i] for i in retrieved_indices]
    sizes = [len(c.split()) for c in chunks]
    return np.mean(sizes) if sizes else 0

def search_tfidf(query, k=5):
    indices, scores = get_similarities_cached(query, k)
    return list(indices), list(scores)

# ===================================
# 4. Ground truth
# ===================================
def find_relevant_chunks_fast(keyword_list, documents, max_chunks=3):
    relevant = []
    keywords_lower = [kw.lower() for kw in keyword_list]
    for i, doc in enumerate(documents):
        if len(relevant) >= max_chunks:
            break
        doc_lower = doc.lower()
        if any(kw in doc_lower for kw in keywords_lower):
            relevant.append(i)
    return relevant

queries = [
    "Who saves Bella from the van?",
    "Which Cullen family member is a doctor?",
]

keywords_per_query = [
    ["edward", "van", "save"],
    ["carlisle", "doctor"],
]

print(f"\n🔍 Consultas definidas: {len(queries)}")
print("\n🤖 Generando ground truth automático...")
ground_truth = []
for i, keywords in enumerate(keywords_per_query):
    relevant = find_relevant_chunks_fast(keywords, documents, max_chunks=3)
    ground_truth.append(relevant)
    print(f"Query {i+1}: {len(relevant)} chunks relevantes")

# ===================================
# 5. Evaluación con generación estilo chat
# ===================================
def evaluate_with_chat(queries, ground_truth, documents, k=5):
    recalls, precisions, context_sizes, all_responses = [], [], [], []

    for i, query in enumerate(queries):
        retrieved_indices, scores = search_tfidf(query, k=k)
        recall = recall_at_k(retrieved_indices, ground_truth[i])
        precision = precision_at_k(retrieved_indices, ground_truth[i], k=k)
        context_size = average_context_size(retrieved_indices, documents)

        recalls.append(recall)
        precisions.append(precision)
        context_sizes.append(context_size)

        context = "\n".join([documents[j] for j in retrieved_indices])
        prompt = f"""
You are a knowledgeable assistant who knows the Twilight Saga.
Answer naturally, briefly, like in a chat.
Use the context below ONLY if it contains the answer.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question: {query}
Answer:
"""
        print
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_answer = response.choices[0].message.content.strip()

        all_responses.append({
            "query": query,
            "retrieved_indices": retrieved_indices,
            "recall": recall,
            "precision": precision,
            "context_size": context_size,
            "answer": generated_answer
        })

    return {
        "recall": np.mean(recalls),
        "precision": np.mean(precisions),
        "context_size": np.mean(context_sizes),
        "details": all_responses,
        "recalls": recalls,
        "precisions": precisions,
        "context_sizes": context_sizes
    }

# Evaluación Modelo Base
results_base = evaluate_with_chat(queries, ground_truth, documents, k=5)
print(results_base['details'])
# ===================================
# 6. Evaluación Modelo B
# ===================================
model_b = RAGModelB(
    preprocessed_base_dir=os.path.join(project_root, "data", "preprocessed"),
    similarity_threshold=0.75
)
model_b.prepare_documents(chunk_config="processed_400_100")

def evaluate_model_b_chat(model, queries, ground_truth, k=5):
    recalls, precisions, context_sizes, all_responses = [], [], [], []
    for i, query in enumerate(queries):
        retrieved = model.query(query, top_k=k)
        retrieved_indices = [idx_meta.get('chunk_number', 0) for _, _, idx_meta in retrieved]

        recall = recall_at_k(retrieved_indices, ground_truth[i])
        precision = precision_at_k(retrieved_indices, ground_truth[i], k=k)
        context_size = np.mean([len(chunk.split()) for chunk, _, _ in retrieved]) if retrieved else 0

        recalls.append(recall)
        precisions.append(precision)
        context_sizes.append(context_size)

        context = "\n".join([chunk for chunk, _, _ in retrieved])
        prompt = f"""
You are a knowledgeable assistant who knows the Twilight Saga.
Answer naturally, briefly, like in a chat.
Use the context below ONLY if it contains the answer.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question: {query}
Answer:
"""
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_answer = response.choices[0].message.content.strip()

        all_responses.append({
            "query": query,
            "retrieved_indices": retrieved_indices,
            "recall": recall,
            "precision": precision,
            "context_size": context_size,
            "answer": generated_answer
        })

    return {
        "recall": np.mean(recalls),
        "precision": np.mean(precisions),
        "context_size": np.mean(context_sizes),
        "details": all_responses,
        "recalls": recalls,
        "precisions": precisions,
        "context_sizes": context_sizes
    }

results_b = evaluate_model_b_chat(model_b, queries, ground_truth, k=5)

# ===================================
# 7. Evaluación Modelo C
# ===================================
model_c = RAGModelC(
    preprocessed_base_dir=os.path.join(project_root, "data", "preprocessed"),
    similarity_threshold=0.85
)
model_c.prepare_documents(chunk_config="processed_400_100")
results_c = evaluate_model_b_chat(model_c, queries, ground_truth, k=5)

# ===================================
# 8. Comparación resultados Base vs B vs C
# ===================================
summary_df = pd.DataFrame({
    "query": [r["query"] for r in results_base["details"]],
    "recall_base": results_base['recalls'],
    "precision_base": results_base['precisions'],
    "recall_model_b": results_b['recalls'],
    "precision_model_b": results_b['precisions'],
    "recall_model_c": results_c['recalls'],
    "precision_model_c": results_c['precisions'],
})
print("\n📊 Comparación de métricas Base vs Modelo B vs Modelo C")
print(summary_df)

# ===================================
# 9. Visualización (igual que antes)
# ===================================
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    
    print("\n📊 Generando visualizaciones...")
    
    query_labels = [f"Q{i+1}" for i in range(len(queries))]
    width = 0.2
    x = np.arange(len(queries))
    
    # Recall
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, results_base['recalls'], width, label='Base', color='#2ecc71')
    ax.bar(x, results_b['recalls'], width, label='Modelo B', color='#3498db')
    ax.bar(x + width, results_c['recalls'], width, label='Modelo C', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.set_ylabel('Recall@5')
    ax.set_ylim([0, 1])
    ax.set_title('Comparación Recall@5 por Consulta', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'data', 'comparison_recall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Precision
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, results_base['precisions'], width, label='Base', color='#2ecc71')
    ax.bar(x, results_b['precisions'], width, label='Modelo B', color='#3498db')
    ax.bar(x + width, results_c['precisions'], width, label='Modelo C', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.set_ylabel('Precision@5')
    ax.set_ylim([0, 1])
    ax.set_title('Comparación Precision@5 por Consulta', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'data', 'comparison_precision.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
except ImportError:
    print("\n matplotlib no disponible. Instalar con: pip install matplotlib")
except Exception as e:
    print(f"\n Error en visualización: {e}")