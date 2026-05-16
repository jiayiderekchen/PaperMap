import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def _build_text(df: pd.DataFrame) -> List[str]:
    titles = df["title"].fillna("").astype(str)
    abstracts = df["abstract"].fillna("").astype(str)
    return (titles + "\n" + abstracts).tolist()


def run() -> None:
    parser = argparse.ArgumentParser(description="Corpus analysis for ICLR papers.")
    parser.add_argument("--input", default="data/papers.parquet")
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--top-terms", type=int, default=50)
    parser.add_argument("--top-keywords", type=int, default=50)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(input_path)
    texts = _build_text(df)

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    counts = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    term_freq = counts.sum(axis=0).A1
    top_term_idx = term_freq.argsort()[::-1][: args.top_terms]
    top_terms = [
        {"term": terms[i], "count": int(term_freq[i])} for i in top_term_idx if term_freq[i] > 0
    ]

    keyword_counter: Counter = Counter()
    if "keywords" in df.columns:
        for keywords in df["keywords"]:
            if keywords is not None:
                # Handle numpy arrays, lists, and strings
                if hasattr(keywords, '__iter__') and not isinstance(keywords, str):
                    keyword_counter.update([str(k).strip().lower() for k in keywords if str(k).strip()])
                elif isinstance(keywords, str) and keywords.strip():
                    keyword_counter.update([k.strip().lower() for k in keywords.split(",") if k.strip()])
    top_keywords = [
        {"keyword": k, "count": c} for k, c in keyword_counter.most_common(args.top_keywords)
    ]

    abstract_lengths = df["abstract"].fillna("").astype(str).str.split().map(len)
    
    # Count keywords properly for Series
    keywords_present = 0
    if "keywords" in df.columns:
        for keywords in df["keywords"]:
            if isinstance(keywords, list) and len(keywords) > 0:
                keywords_present += 1
            elif isinstance(keywords, str) and keywords.strip():
                keywords_present += 1
    
    stats = {
        "paper_count": int(len(df)),
        "avg_abstract_tokens": float(abstract_lengths.mean() if len(df) else 0),
        "median_abstract_tokens": float(abstract_lengths.median() if len(df) else 0),
        "keywords_present": keywords_present,
    }

    with (out_dir / "corpus_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=True)
    pd.DataFrame(top_terms).to_csv(out_dir / "top_terms.csv", index=False)
    pd.DataFrame(top_keywords).to_csv(out_dir / "top_keywords.csv", index=False)

    print(f"Wrote corpus analysis to {out_dir}")


if __name__ == "__main__":
    run()
