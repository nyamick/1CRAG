# -*- coding: utf-8 -*-
"""
Эксперимент: сравнение стратегий retrieval для RAG-системы по документации
1С:Предприятие.

Методы:
  1. BM25-подобный TF-IDF baseline (косинусная близость TF-IDF векторов)
  2. LSA (усеченное SVD над TF-IDF матрицей) как "плотные" эмбеддинги,
     обученные локально на корпусе (без внешних предобученных моделей)
  3. Гибрид: Reciprocal Rank Fusion (RRF) объединения ранжирований (1) и (2)

Метрики: Recall@1, Recall@3, Recall@5, MRR (Mean Reciprocal Rank).

Фиксирован random_state для воспроизводимости (см. seed в config).
"""
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
np.random.seed(SEED)

RUSSIAN_STOPWORDS = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
]


def load_data():
    with open("/home/claude/project/data/corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)
    with open("/home/claude/project/data/qa.json", encoding="utf-8") as f:
        qa = json.load(f)
    return corpus, qa


def build_tfidf(corpus_texts):
    vec = TfidfVectorizer(stop_words=RUSSIAN_STOPWORDS, ngram_range=(1, 2), min_df=1)
    doc_matrix = vec.fit_transform(corpus_texts)
    return vec, doc_matrix


def build_lsa(doc_matrix, n_components=10, seed=SEED):
    n_components = min(n_components, doc_matrix.shape[0] - 1, doc_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    doc_lsa = svd.fit_transform(doc_matrix)
    return svd, doc_lsa


def rank_by_scores(scores, doc_ids):
    order = np.argsort(-scores)
    return [doc_ids[i] for i in order]


def rrf_fuse(rank_a, rank_b, k=60):
    """Reciprocal Rank Fusion объединяет два ранжирования списков doc_id."""
    scores = {}
    for rank_list in (rank_a, rank_b):
        for pos, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    fused = sorted(scores.keys(), key=lambda d: -scores[d])
    return fused


def evaluate(method_name, rankings, qa, ks=(1, 3, 5)):
    """rankings: список списков doc_id (ранжирование для каждого вопроса)."""
    n = len(qa)
    recall_at_k = {k: 0 for k in ks}
    reciprocal_ranks = []

    for item, ranking in zip(qa, rankings):
        gold = set(item["gold"])
        rr = 0.0
        for pos, doc_id in enumerate(ranking):
            if doc_id in gold:
                rr = 1.0 / (pos + 1)
                break
        reciprocal_ranks.append(rr)
        for k in ks:
            topk = set(ranking[:k])
            if topk & gold:
                recall_at_k[k] += 1

    result = {"method": method_name, "n_queries": n,
              "MRR": round(float(np.mean(reciprocal_ranks)), 4)}
    for k in ks:
        result[f"Recall@{k}"] = round(recall_at_k[k] / n, 4)
    return result


def main():
    corpus, qa = load_data()
    doc_ids = [d["id"] for d in corpus]
    corpus_texts = [d["title"] + ". " + d["text"] for d in corpus]

    vec, doc_tfidf = build_tfidf(corpus_texts)
    svd, doc_lsa = build_lsa(doc_tfidf, n_components=10, seed=SEED)

    tfidf_rankings, lsa_rankings, hybrid_rankings = [], [], []
    latencies = {"tfidf": [], "lsa": [], "hybrid": []}

    for item in qa:
        query = item["q"]

        t0 = time.perf_counter()
        q_tfidf = vec.transform([query])
        scores_tfidf = cosine_similarity(q_tfidf, doc_tfidf)[0]
        rank_tfidf = rank_by_scores(scores_tfidf, doc_ids)
        latencies["tfidf"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        q_lsa = svd.transform(q_tfidf)
        scores_lsa = cosine_similarity(q_lsa, doc_lsa)[0]
        rank_lsa = rank_by_scores(scores_lsa, doc_ids)
        latencies["lsa"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rank_hybrid = rrf_fuse(rank_tfidf, rank_lsa)
        latencies["hybrid"].append(time.perf_counter() - t0 + latencies["tfidf"][-1] + latencies["lsa"][-1])

        tfidf_rankings.append(rank_tfidf)
        lsa_rankings.append(rank_lsa)
        hybrid_rankings.append(rank_hybrid)

    results = [
        evaluate("TF-IDF (baseline)", tfidf_rankings, qa),
        evaluate("LSA (SVD dense embeddings)", lsa_rankings, qa),
        evaluate("Hybrid (RRF fusion)", hybrid_rankings, qa),
    ]

    for r, name in zip(results, latencies):
        r["avg_latency_ms"] = round(float(np.mean(latencies[name])) * 1000, 3)

    # Сохраняем результаты
    with open("/home/claude/project/results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Печатаем таблицу
    print(f"{'Метод':<28}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>8}{'Latency(ms)':>13}")
    for r in results:
        print(f"{r['method']:<28}{r['Recall@1']:>10}{r['Recall@3']:>10}{r['Recall@5']:>10}{r['MRR']:>8}{r['avg_latency_ms']:>13}")

    # Сохраняем детальные ранжирования для приложения/отладки
    detail = []
    for item, rt, rl, rh in zip(qa, tfidf_rankings, lsa_rankings, hybrid_rankings):
        detail.append({
            "question": item["q"], "gold": item["gold"],
            "top3_tfidf": rt[:3], "top3_lsa": rl[:3], "top3_hybrid": rh[:3],
        })
    with open("/home/claude/project/results/detailed_rankings.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    print("\nРезультаты сохранены в results/metrics.json и results/detailed_rankings.json")


if __name__ == "__main__":
    main()
