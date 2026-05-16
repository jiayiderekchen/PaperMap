import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def _build_text(df: pd.DataFrame) -> List[str]:
    titles = df["title"].fillna("").astype(str)
    abstracts = df["abstract"].fillna("").astype(str)
    return (titles + "\n" + abstracts).tolist()


def _top_terms_for_cluster(
    tfidf_matrix,
    feature_names: List[str],
    indices: np.ndarray,
    top_k: int,
) -> List[str]:
    if indices.size == 0:
        return []
    mean_vector = tfidf_matrix[indices].mean(axis=0)
    mean_dense = np.asarray(mean_vector).ravel()
    top_idx = mean_dense.argsort()[::-1][:top_k]
    return [feature_names[i] for i in top_idx if mean_dense[i] > 0]


def run() -> None:
    parser = argparse.ArgumentParser(description="Cluster ICLR papers using TF-IDF + SVD + KMeans.")
    parser.add_argument("--input", default="data/papers.parquet")
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--svd-dims", type=int, default=100)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(input_path)
    texts = _build_text(df)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    svd = TruncatedSVD(n_components=min(args.svd_dims, tfidf_matrix.shape[1] - 1))
    embeddings = svd.fit_transform(tfidf_matrix)

    kmeans = MiniBatchKMeans(n_clusters=args.clusters, random_state=42, batch_size=256)
    cluster_ids = kmeans.fit_predict(embeddings)

    cluster_labels: Dict[int, str] = {}
    cluster_summaries: List[Dict[str, object]] = []
    for cluster_id in range(args.clusters):
        indices = np.where(cluster_ids == cluster_id)[0]
        terms = _top_terms_for_cluster(tfidf_matrix, feature_names, indices, args.top_terms)
        label = ", ".join(terms[:3]) if terms else f"Cluster {cluster_id}"
        cluster_labels[cluster_id] = label
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "size": int(indices.size),
                "top_terms": terms,
            }
        )

    df_out = df.copy()
    df_out["cluster_id"] = cluster_ids
    df_out["cluster_label"] = df_out["cluster_id"].map(cluster_labels)

    (out_dir / "papers_with_clusters.parquet").parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_dir / "papers_with_clusters.parquet", index=False)
    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(cluster_summaries, handle, indent=2, ensure_ascii=True)
    with (out_dir / "paper_index.json").open("w", encoding="utf-8") as handle:
        json.dump(df_out["paper_id"].tolist(), handle, indent=2, ensure_ascii=True)

    print(f"Wrote {len(df_out)} clustered papers to {out_dir}")


if __name__ == "__main__":
    run()
