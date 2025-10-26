# 🧩 redundancy.py — reducción de redundancia y selección de chunks relevantes

from sentence_transformers import SentenceTransformer, util
import numpy as np

# 🔹 Cargamos el modelo una sola vez (para que no se recargue en cada consulta)
model = SentenceTransformer('all-MiniLM-L6-v2')


def remove_duplicates(chunks):
    """
    Elimina chunks idénticos o casi idénticos (por texto exacto o en minúsculas).
    Retorna una lista de chunks únicos.
    """
    unique = []
    seen = set()

    for c in chunks:
        norm = c.strip().lower()
        if norm not in seen:
            unique.append(c)
            seen.add(norm)

    return unique


def mmr_selection(query, chunks, top_k=5, lambda_param=0.5):
    """
    Selecciona los chunks más relevantes y diversos usando MMR (Maximal Marginal Relevance).

    Parámetros:
    - query: texto de la consulta.
    - chunks: lista de textos.
    - top_k: número de chunks a seleccionar.
    - lambda_param: balance entre relevancia (1) y diversidad (0).

    Retorna:
    - Lista de chunks seleccionados.
    """
    if not chunks:
        return []

    # Asegurar que top_k no supere el número de chunks disponibles
    top_k = min(top_k, len(chunks))

    # Codificamos consulta y documentos
    query_emb = model.encode([query], convert_to_tensor=True)
    doc_embs = model.encode(chunks, convert_to_tensor=True)

    # Similaridad entre query y documentos
    sim_q_d = util.pytorch_cos_sim(query_emb, doc_embs)[0]

    # Primer chunk más relevante
    selected_idx = [int(np.argmax(sim_q_d))]

    # Iteramos para seleccionar los siguientes chunks maximizando la diversidad
    for _ in range(top_k - 1):
        remaining = list(set(range(len(chunks))) - set(selected_idx))
        mmr_scores = []

        for i in remaining:
            # Diversidad: similitud máxima con los ya seleccionados
            diversity = max(util.pytorch_cos_sim(doc_embs[i], doc_embs[selected_idx]).cpu().numpy())
            # Balance entre relevancia y diversidad
            score = lambda_param * float(sim_q_d[i].cpu().numpy()) - (1 - lambda_param) * diversity
            mmr_scores.append(score)

        if not mmr_scores:
            break

        # Seleccionar el índice con el puntaje MMR más alto
        idx = remaining[int(np.argmax(mmr_scores))]
        selected_idx.append(idx)

    # Retornar los chunks seleccionados
    selected_chunks = [chunks[i] for i in selected_idx]
    return selected_chunks
