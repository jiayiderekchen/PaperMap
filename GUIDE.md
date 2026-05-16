# Paper Crawler — User Guide

A unified tool for crawling, analysing, and exploring accepted papers from major ML conferences (NeurIPS, ICLR, ICML, AAAI).

---

## Quick Start — Launch the App

```bash
cd /Users/jiayichen/Files/paper_crawler
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

Use the sidebar to pick a **Conference** and **Year**. Each combination has two views:

| View | What it shows |
|---|---|
| **Overview** | Stats, research topics, paper landscape (2D map), keywords, clusters |
| **Browse Papers** | Full-text search, topic/cluster filters, paper details, related papers |

---

## Project Structure

```
paper_crawler/
├── app/
│   └── streamlit_app.py        # Unified Streamlit UI
├── crawler/
│   ├── run_crawl.py            # OpenReview crawler (NeurIPS, ICLR, ICML)
│   ├── crawl_aaai_ojs.py       # AAAI OJS CLI entrypoint
│   └── aaai_ojs_client.py      # AAAI web scraper
├── analysis/
│   ├── cluster_analysis.py     # TF-IDF + MiniBatchKMeans clustering
│   ├── corpus_analysis.py      # Stats, keywords, embeddings
│   └── topic_extraction.py     # High-level topic & subtopic labelling
├── neurips2025/                 # NeurIPS 2025 data & analysis
├── iclr2026/                   # ICLR 2026 data & analysis
├── icml2025/                   # ICML 2025 data & analysis
├── aaai2026/                   # AAAI 2026 data & analysis
│   ├── data/
│   │   ├── papers.parquet
│   │   └── papers.jsonl
│   └── analysis/
│       ├── papers_with_clusters.parquet
│       ├── embeddings.npy
│       ├── cluster_summary.json
│       ├── topic_hierarchy.json
│       ├── paper_topics.json
│       └── top_keywords.csv
└── requirements.txt
```

Each conference/year follows the same `{conf}{year}/data/` + `{conf}{year}/analysis/` layout.

---

## Setup (first time)

```bash
cd /Users/jiayichen/Files/paper_crawler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Adding a New Conference Year

### Option A — OpenReview conferences (NeurIPS, ICLR, ICML)

```bash
source .venv/bin/activate

# 1. Crawl papers
python crawler/run_crawl.py \
    --conference ICLR \
    --year 2025 \
    --out-dir iclr2025/data \
    --sleep-s 2 --page-sleep 2 --max-retries 5

# 2. Run analysis pipeline
python analysis/cluster_analysis.py  --input iclr2025/data/papers.parquet --out-dir iclr2025/analysis
python analysis/corpus_analysis.py   --input iclr2025/data/papers.parquet --out-dir iclr2025/analysis
python analysis/topic_extraction.py  --input iclr2025/data/papers.parquet --out-dir iclr2025/analysis
```

Or use the pre-built shell script (e.g. `bash icml2026/crawl_icml.sh`).

### Option B — AAAI (scraped from ojs.aaai.org)

```bash
source .venv/bin/activate

# 1. Crawl papers (~20 min for a full year)
python crawler/crawl_aaai_ojs.py --year 2026 --out-dir aaai2026/data

# Resume if interrupted (merges with existing data automatically)
python crawler/crawl_aaai_ojs.py --year 2026 --out-dir aaai2026/data --start-issue 724

# 2. Run analysis pipeline (same as above)
python analysis/cluster_analysis.py  --input aaai2026/data/papers.parquet --out-dir aaai2026/analysis
python analysis/corpus_analysis.py   --input aaai2026/data/papers.parquet --out-dir aaai2026/analysis
python analysis/topic_extraction.py  --input aaai2026/data/papers.parquet --out-dir aaai2026/analysis
```

Or use the pre-built shell script: `bash aaai2026/crawl_aaai.sh`

### Register the new year in the app

Open `app/streamlit_app.py` and ensure the year is in the `YEARS` list (top of file). If data is not yet available, add it to `_EXCLUDED_YEARS` to hide it from the dropdown until ready.

---

## UI Features

### Overview page

- **Stat cards** — total papers, largest topic, papers in that topic
- **Research Topics** — click any bar to drill down; the keywords chart, scatter highlight, and paper list all update to reflect your selection
- **Top Keywords** — updates dynamically when a topic is selected
- **Paper Landscape** — 2D PCA map of all papers coloured by cluster; hover to preview, click to open paper details
- **Drill-down panel** (appears after clicking a topic):
  - Subtopics bar chart — click to filter further
  - Cluster breakdown — shows clusters for the selected topic or subtopic
  - Paper list

### Browse Papers page

- Full-text search across titles and abstracts
- Filter by topic, subtopic, and cluster
- Click any paper to see abstract, authors, and a "Related Papers" list (cosine similarity on embeddings)

---

## Data Notes

| Conference | Source | Keywords available |
|---|---|---|
| NeurIPS | OpenReview API | Yes |
| ICLR | OpenReview API | Yes |
| ICML | OpenReview API | Yes |
| AAAI | ojs.aaai.org (web scrape) | No (not published on OJS) |

AAAI papers do not include author-submitted keywords because the OJS proceedings site does not expose them. Topic and cluster analysis still works via abstract text.
