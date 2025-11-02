# ============================================================
# NOTEBOOK COMPLETO: Evaluación RAG con respuestas estilo chat
# ============================================================
# ============================================================
# EVALUACIÓN RAG CORREGIDA
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
print("✅ Project root:", project_root)
# ===================================
# 1. CARGAR DATOS
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

# Crear mapeo de texto -> índice para comparación consistente
text_to_index = {text: idx for idx, text in enumerate(documents)}

print(f"📄 Total chunks: {len(documents)}")

# ===================================
# 2. ÍNDICE TF-IDF MODELO BASE
# ===================================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(documents)
print(f"✅ Índice TF-IDF: {X.shape}")

# ===================================
# 3. GROUND TRUTH CORRECTO
# ===================================
def find_relevant_chunks_corrected(keyword_list, documents, max_chunks=5):
    """Busca en TODOS los documentos, no solo los primeros N"""
    relevant = []
    keywords_lower = [kw.lower() for kw in keyword_list]
    
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        # Verificar que contenga TODAS las keywords (más estricto)
        if all(kw in doc_lower for kw in keywords_lower):
            relevant.append(i)
    
    # Si no hay matches con ALL, buscar con ANY
    if not relevant:
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            if any(kw in doc_lower for kw in keywords_lower):
                relevant.append(i)
                if len(relevant) >= max_chunks:
                    break
    
    return relevant[:max_chunks]

queries = [
    "Who saves Bella from the van?",
    "Which Cullen family member is a doctor?",
]

keywords_per_query = [
    ["edward", "van", "save"],
    ["carlisle", "doctor"],
]

print("\n🔍 Generando Ground Truth CORREGIDO...")
ground_truth = []
for i, keywords in enumerate(keywords_per_query):
    relevant = find_relevant_chunks_corrected(keywords, documents, max_chunks=5)
    ground_truth.append(relevant)
    print(f"Query {i+1}: {len(relevant)} chunks relevantes")
    if relevant:
        print(f"  Índices: {relevant[:3]}...")
        print(f"  Ejemplo: {documents[relevant[0]][:100]}...")

# ===================================
# 4. FUNCIONES DE EVALUACIÓN
# ===================================
def recall_at_k(retrieved_indices, relevant_indices):
    if not relevant_indices:
        return 0.0
    hits = len(set(retrieved_indices) & set(relevant_indices))
    return hits / len(relevant_indices)

def precision_at_k(retrieved_indices, relevant_indices, k=5):
    if not relevant_indices:
        return 0.0
    hits = len(set(retrieved_indices) & set(relevant_indices))
    return hits / k

def mrr(retrieved_indices, relevant_indices):
    """Mean Reciprocal Rank"""
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0

@lru_cache(maxsize=128)
def search_tfidf_cached(query, k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return tuple(top_k_indices), tuple(similarities[top_k_indices])

def search_tfidf(query, k=5):
    indices, scores = search_tfidf_cached(query, k)
    return list(indices), list(scores)

# ===================================
# 5. GENERACIÓN DE RESPUESTAS
# ===================================
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
# 6. EVALUACIÓN MODELO BASE (TF-IDF)
# ===================================
def evaluate_base_model(queries, ground_truth, documents, k=5):
    """Evaluación del modelo base con TF-IDF"""
    results = []
    
    for i, query in enumerate(queries):
        retrieved_indices, scores = search_tfidf(query, k=k)
        
        # Métricas
        recall = recall_at_k(retrieved_indices, ground_truth[i])
        precision = precision_at_k(retrieved_indices, ground_truth[i], k=k)
        mrr_score = mrr(retrieved_indices, ground_truth[i])
        
        # Generar respuesta
        context = "\n".join([documents[j] for j in retrieved_indices])
        answer = generate_answer(context, query)
        
        results.append({
            "query": query,
            "retrieved_indices": retrieved_indices,
            "recall": recall,
            "precision": precision,
            "mrr": mrr_score,
            "answer": answer
        })
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Recall@{k}: {recall:.3f} | Precision@{k}: {precision:.3f} | MRR: {mrr_score:.3f}")
        print(f"Answer: {answer}")
    
    return results

# ===================================
# 7. EVALUACIÓN BASELINE (BM25)
# ===================================
def evaluate_baseline(model, queries, ground_truth, k=5):
    """Evaluación específica para RAGBaseline (devuelve dict con 'results')"""
    results = []
    
    for i, query in enumerate(queries):
        result = model.query(query, top_k=k, show_details=False)
        retrieved_data = result["results"]
        retrieved_texts = [chunk for chunk, _, _ in retrieved_data]
        
        # Mapear textos a índices del array base
        retrieved_indices = []
        for text in retrieved_texts:
            if text in text_to_index:
                retrieved_indices.append(text_to_index[text])
            else:
                for doc_idx, doc in enumerate(documents):
                    if text[:200] in doc or doc[:200] in text:
                        retrieved_indices.append(doc_idx)
                        break
        
        # Métricas
        recall = recall_at_k(retrieved_indices, ground_truth[i])
        precision = precision_at_k(retrieved_indices, ground_truth[i], k=k)
        mrr_score = mrr(retrieved_indices, ground_truth[i])
        
        # Generar respuesta
        context = "\n".join(retrieved_texts)
        answer = generate_answer(context, query)
        
        results.append({
            "query": query,
            "retrieved_indices": retrieved_indices,
            "recall": recall,
            "precision": precision,
            "mrr": mrr_score,
            "answer": answer
        })
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Recall@{k}: {recall:.3f} | Precision@{k}: {precision:.3f} | MRR: {mrr_score:.3f}")
        print(f"Answer: {answer}")
    
    return results

# ===================================
# 8. EVALUACIÓN MODEL A (FAISS)
# ===================================
def evaluate_model_a(model, queries, ground_truth, k=5):
    """Evaluación específica para RAGModelA (FAISS)"""
    results = []
    
    for i, query in enumerate(queries):
        # RAGModelA.query retorna lista de dicts
        retrieved = model.query(query, top_k=k, show_details=False)
        retrieved_texts = [r['text'] for r in retrieved]
        
        # Mapear textos a índices del array base
        retrieved_indices = []
        for text in retrieved_texts:
            if text in text_to_index:
                retrieved_indices.append(text_to_index[text])
            else:
                for doc_idx, doc in enumerate(documents):
                    if text[:200] in doc or doc[:200] in text:
                        retrieved_indices.append(doc_idx)
                        break
        
        # Métricas
        recall = recall_at_k(retrieved_indices, ground_truth[i])
        precision = precision_at_k(retrieved_indices, ground_truth[i], k=k)
        mrr_score = mrr(retrieved_indices, ground_truth[i])
        
        # Generar respuesta
        context = "\n".join(retrieved_texts)
        answer = generate_answer(context, query)
        
        results.append({
            "query": query,
            "retrieved_indices": retrieved_indices,
            "recall": recall,
            "precision": precision,
            "mrr": mrr_score,
            "answer": answer
        })
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Recall@{k}: {recall:.3f} | Precision@{k}: {precision:.3f} | MRR: {mrr_score:.3f}")
        print(f"Answer: {answer}")
    
    return results

# ===================================
# 9. EVALUACIÓN MODEL B/C (TF-IDF variants)
# ===================================
def evaluate_tfidf_model(model, queries, ground_truth, k=5):
    """Evaluación para RAGModelB y RAGModelC (devuelven lista de tuplas)"""
    results = []
    
    for i, query in enumerate(queries):
        retrieved = model.query(query, top_k=k)
        retrieved_texts = [chunk for chunk, _, _ in retrieved]
        
        # Mapear textos a índices del array base
        retrieved_indices = []
        for text in retrieved_texts:
            if text in text_to_index:
                retrieved_indices.append(text_to_index[text])
            else:
                for doc_idx, doc in enumerate(documents):
                    if text[:200] in doc or doc[:200] in text:
                        retrieved_indices.append(doc_idx)
                        break
        
        # Métricas
        recall = recall_at_k(retrieved_indices, ground_truth[i])
        precision = precision_at_k(retrieved_indices, ground_truth[i], k=k)
        mrr_score = mrr(retrieved_indices, ground_truth[i])
        
        # Generar respuesta
        context = "\n".join(retrieved_texts)
        answer = generate_answer(context, query)
        
        results.append({
            "query": query,
            "retrieved_indices": retrieved_indices,
            "recall": recall,
            "precision": precision,
            "mrr": mrr_score,
            "answer": answer
        })
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Recall@{k}: {recall:.3f} | Precision@{k}: {precision:.3f} | MRR: {mrr_score:.3f}")
        print(f"Answer: {answer}")
    
    return results

# ===================================
# 10. EJECUTAR EVALUACIONES
# ===================================

# Modelo Base (TF-IDF simple)
print("\n" + "="*80)
print("EVALUANDO MODELO BASE (TF-IDF)")
print("="*80)
results_base = evaluate_base_model(queries, ground_truth, documents, k=5)

# Baseline (BM25)
print("\n" + "="*80)
print("EVALUANDO BASELINE (BM25)")
print("="*80)
baseline = RAGBaseline(
    preprocessed_base_dir=BASE_PREPROCESSED,
    k1=1.5,
    b=0.75
)
baseline.prepare_documents(chunk_config="processed_400_100")
results_baseline = evaluate_baseline(baseline, queries, ground_truth, k=5)

# Model A (FAISS + Embeddings)
print("\n" + "="*80)
print("EVALUANDO MODEL A (FAISS + EMBEDDINGS)")
print("="*80)
try:
    model_a = RAGModelA(
        preprocessed_base_dir=BASE_PREPROCESSED,
        use_gpu=False
    )
    model_a.prepare_documents(chunk_config="processed_400_100", index_type="flat", batch_size=32)
    results_a = evaluate_model_a(model_a, queries, ground_truth, k=5)
except Exception as e:
    print(f"⚠️ Error en Model A: {e}")
    print("Saltando Model A (puede requerir sentence-transformers y faiss)")
    results_a = None

# Model B (TF-IDF + Dedup threshold=0.75)
print("\n" + "="*80)
print("EVALUANDO MODEL B (TF-IDF + DEDUP 0.75)")
print("="*80)
model_b = RAGModelB(
    preprocessed_base_dir=BASE_PREPROCESSED,
    similarity_threshold=0.75
)
model_b.prepare_documents(chunk_config="processed_400_100")
results_b = evaluate_tfidf_model(model_b, queries, ground_truth, k=5)

# Model C (TF-IDF + Dedup threshold=0.85)
print("\n" + "="*80)
print("EVALUANDO MODEL C (TF-IDF + DEDUP 0.85)")
print("="*80)
model_c = RAGModelC(
    preprocessed_base_dir=BASE_PREPROCESSED,
    similarity_threshold=0.85
)
model_c.prepare_documents(chunk_config="processed_400_100")
results_c = evaluate_tfidf_model(model_c, queries, ground_truth, k=5)

# ===================================
# 11. COMPARACIÓN FINAL
# ===================================
print("\n" + "="*80)
print("COMPARACIÓN FINAL")
print("="*80)

# Crear DataFrame de comparación
comparison_data = {
    "Query": [r["query"] for r in results_base],
    "Recall_Base": [r["recall"] for r in results_base],
    "Recall_Baseline": [r["recall"] for r in results_baseline],
    "Recall_B": [r["recall"] for r in results_b],
    "Recall_C": [r["recall"] for r in results_c],
    "Precision_Base": [r["precision"] for r in results_base],
    "Precision_Baseline": [r["precision"] for r in results_baseline],
    "Precision_B": [r["precision"] for r in results_b],
    "Precision_C": [r["precision"] for r in results_c],
    "MRR_Base": [r["mrr"] for r in results_base],
    "MRR_Baseline": [r["mrr"] for r in results_baseline],
    "MRR_B": [r["mrr"] for r in results_b],
    "MRR_C": [r["mrr"] for r in results_c],
}

# Agregar Model A si está disponible
if results_a:
    comparison_data["Recall_A"] = [r["recall"] for r in results_a]
    comparison_data["Precision_A"] = [r["precision"] for r in results_a]
    comparison_data["MRR_A"] = [r["mrr"] for r in results_a]

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Promedios
print("\n" + "="*80)
print("PROMEDIOS")
print("="*80)

models_list = ['Base', 'Baseline', 'Model A', 'Model B', 'Model C'] if results_a else ['Base', 'Baseline', 'Model B', 'Model C']
print(f"{'Metric':<15} " + " ".join([f"{m:<12}" for m in models_list]))
print("-" * (15 + 13 * len(models_list)))

# Recall
recall_means = [
    comparison_df['Recall_Base'].mean(),
    comparison_df['Recall_Baseline'].mean(),
]
if results_a:
    recall_means.append(comparison_df['Recall_A'].mean())
recall_means.extend([
    comparison_df['Recall_B'].mean(),
    comparison_df['Recall_C'].mean()
])
print(f"{'Recall@5':<15} " + " ".join([f"{m:.3f}        " for m in recall_means]))

# Precision
precision_means = [
    comparison_df['Precision_Base'].mean(),
    comparison_df['Precision_Baseline'].mean(),
]
if results_a:
    precision_means.append(comparison_df['Precision_A'].mean())
precision_means.extend([
    comparison_df['Precision_B'].mean(),
    comparison_df['Precision_C'].mean()
])
print(f"{'Precision@5':<15} " + " ".join([f"{m:.3f}        " for m in precision_means]))

# MRR
mrr_means = [
    comparison_df['MRR_Base'].mean(),
    comparison_df['MRR_Baseline'].mean(),
]
if results_a:
    mrr_means.append(comparison_df['MRR_A'].mean())
mrr_means.extend([
    comparison_df['MRR_B'].mean(),
    comparison_df['MRR_C'].mean()
])
print(f"{'MRR':<15} " + " ".join([f"{m:.3f}        " for m in mrr_means]))

# ===================================
# 12. MOSTRAR RESPUESTAS GENERADAS
# ===================================
print("\n" + "="*80)
print("RESPUESTAS GENERADAS")
print("="*80)

for idx, query in enumerate(queries):
    print(f"\n🧠 Query {idx+1}: {query}")
    print("-" * 80)
    print(f"Base:     {results_base[idx]['answer']}")
    print(f"Baseline: {results_baseline[idx]['answer']}")
    if results_a:
        print(f"Model A:  {results_a[idx]['answer']}")
    print(f"Model B:  {results_b[idx]['answer']}")
    print(f"Model C:  {results_c[idx]['answer']}")
    print("="*80)

# ===================================
# 13. VISUALIZACIÓN
# ===================================
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    
    print("\n📊 Generando visualizaciones...")
    
    query_labels = [f"Q{i+1}" for i in range(len(queries))]
    n_models = 5 if results_a else 4
    width = 0.15
    x = np.arange(len(queries))
    
    colors = {
        'Base': '#2ecc71',
        'Baseline': '#f39c12',
        'Model A': '#3498db',
        'Model B': '#9b59b6',
        'Model C': '#e74c3c'
    }
    
    # Gráfica 1: Recall
    fig, ax = plt.subplots(figsize=(12,6))
    offset = -width * (n_models//2)
    
    ax.bar(x + offset, comparison_df['Recall_Base'], width, label='Base (TF-IDF)', color=colors['Base'])
    offset += width
    ax.bar(x + offset, comparison_df['Recall_Baseline'], width, label='Baseline (BM25)', color=colors['Baseline'])
    offset += width
    
    if results_a:
        ax.bar(x + offset, comparison_df['Recall_A'], width, label='Model A (FAISS)', color=colors['Model A'])
        offset += width
    
    ax.bar(x + offset, comparison_df['Recall_B'], width, label='Model B (Dedup 0.75)', color=colors['Model B'])
    offset += width
    ax.bar(x + offset, comparison_df['Recall_C'], width, label='Model C (Dedup 0.85)', color=colors['Model C'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.set_ylabel('Recall@5')
    ax.set_ylim([0, 1])
    ax.set_title('Comparación Recall@5 por Consulta', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'data', 'comparison_recall.png'), dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: comparison_recall.png")
    plt.show()
    
    # Gráfica 2: Precision
    fig, ax = plt.subplots(figsize=(12,6))
    offset = -width * (n_models//2)
    
    ax.bar(x + offset, comparison_df['Precision_Base'], width, label='Base (TF-IDF)', color=colors['Base'])
    offset += width
    ax.bar(x + offset, comparison_df['Precision_Baseline'], width, label='Baseline (BM25)', color=colors['Baseline'])
    offset += width
    
    if results_a:
        ax.bar(x + offset, comparison_df['Precision_A'], width, label='Model A (FAISS)', color=colors['Model A'])
        offset += width
    
    ax.bar(x + offset, comparison_df['Precision_B'], width, label='Model B (Dedup 0.75)', color=colors['Model B'])
    offset += width
    ax.bar(x + offset, comparison_df['Precision_C'], width, label='Model C (Dedup 0.85)', color=colors['Model C'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.set_ylabel('Precision@5')
    ax.set_ylim([0, 1])
    ax.set_title('Comparación Precision@5 por Consulta', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'data', 'comparison_precision.png'), dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: comparison_precision.png")
    plt.show()
    
    # Gráfica 3: Resumen de promedios
    fig, ax = plt.subplots(figsize=(14,7))
    metrics = ['Recall@5', 'Precision@5', 'MRR']
    
    x_pos = np.arange(len(metrics))
    width = 0.15
    offset = -width * (n_models//2)
    
    ax.bar(x_pos + offset, recall_means[:1] + precision_means[:1] + mrr_means[:1], 
           width, label='Base (TF-IDF)', color=colors['Base'])
    offset += width
    
    ax.bar(x_pos + offset, [recall_means[1], precision_means[1], mrr_means[1]], 
           width, label='Baseline (BM25)', color=colors['Baseline'])
    offset += width
    
    if results_a:
        ax.bar(x_pos + offset, [recall_means[2], precision_means[2], mrr_means[2]], 
               width, label='Model A (FAISS)', color=colors['Model A'])
        offset += width
    
    b_idx = 2 if results_a else 1
    ax.bar(x_pos + offset, [recall_means[b_idx+1], precision_means[b_idx+1], mrr_means[b_idx+1]], 
           width, label='Model B (Dedup 0.75)', color=colors['Model B'])
    offset += width
    
    ax.bar(x_pos + offset, [recall_means[-1], precision_means[-1], mrr_means[-1]], 
           width, label='Model C (Dedup 0.85)', color=colors['Model C'])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.set_title('Comparación de Métricas Promedio - Todos los Modelos', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'data', 'comparison_summary.png'), dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: comparison_summary.png")
    plt.show()
    
    print("\n✅ Todas las visualizaciones generadas exitosamente")
    
except ImportError:
    print("\n⚠️ matplotlib no disponible. Instalar con: pip install matplotlib")
except Exception as e:
    print(f"\n❌ Error en visualización: {e}")

# ===================================
# 14. GUARDAR RESULTADOS
# ===================================
print("\n" + "="*80)
print("GUARDANDO RESULTADOS")
print("="*80)

results_summary = {
    'queries': queries,
    'ground_truth': ground_truth,
    'base': results_base,
    'baseline': results_baseline,
    'model_b': results_b,
    'model_c': results_c,
    'comparison': comparison_df.to_dict()
}

if results_a:
    results_summary['model_a'] = results_a

import json
results_path = os.path.join(project_root, 'data', 'evaluation_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json.dump(results_summary, f, indent=2, ensure_ascii=False, default=convert_to_serializable)

print(f"✅ Resultados guardados en: {results_path}")

csv_path = os.path.join(project_root, 'data', 'evaluation_comparison.csv')
comparison_df.to_csv(csv_path, index=False)
print(f"✅ Comparación guardada en: {csv_path}")

print("\n" + "="*80)
print("🎉 EVALUACIÓN COMPLETA")
print("="*80)
print(f"Modelos evaluados: {len(models_list)}")
print(f"Queries evaluadas: {len(queries)}")
print(f"Archivos generados:")
print(f"  • evaluation_results.json")
print(f"  • evaluation_comparison.csv")
print(f"  • comparison_recall.png")
print(f"  • comparison_precision.png")
print(f"  • comparison_summary.png")
print("="*80)