# Paper Crawler - Multi-Conference System

This repository contains complete paper analysis systems for multiple AI conferences.

## Available Conferences

### 🔵 ICLR 2026 
- **Papers**: 5,359 accepted papers
- **UI Port**: 8501
- **Directory**: `./` (root)
- **Launch**: `streamlit run app/streamlit_app.py`
- **URL**: http://localhost:8501

### 🔴 NeurIPS 2025
- **Papers**: 5,286 accepted papers  
- **UI Port**: 8502
- **Directory**: `./neurips2025/`
- **Launch**: `streamlit run neurips2025/app/streamlit_app.py --server.port 8502`
- **URL**: http://localhost:8502

## Quick Start

### Run Both UIs Simultaneously
```bash
# Terminal 1: ICLR 2026
source .venv/bin/activate
streamlit run app/streamlit_app.py

# Terminal 2: NeurIPS 2025
source .venv/bin/activate
streamlit run neurips2025/app/streamlit_app.py --server.port 8502
```

### Crawl New Conference
```bash
source .venv/bin/activate

# For ICLR
python crawler/run_crawl.py --conference ICLR --year 2026 --out-dir data

# For NeurIPS
python crawler/run_crawl.py --conference NeurIPS --year 2025 --out-dir neurips2025/data
```

## Features

### 🎯 Hierarchical Topic Browsing
- 13 major research topics
- 50+ subtopics per conference
- Automatic paper categorization

### 🔬 Automatic Clustering
- 50 clusters per conference
- TF-IDF + SVD embeddings
- Unsupervised topic discovery

### 🔍 Multi-Filter Search
- Full-text search
- Topic/subtopic filtering
- Cluster filtering
- Combined filters

### 📊 Paper Details
- Title, abstract, authors, keywords
- Topic assignments
- Direct links (arXiv, PDF, OpenReview)
- Related papers via similarity

## Research Insights

### ICLR 2026 vs NeurIPS 2025

| Topic | ICLR 2026 | NeurIPS 2025 |
|-------|-----------|--------------|
| **Large Language Models** | 2,162 (40%) | 1,619 (31%) |
| **NLP** | 1,696 (32%) | 1,322 (25%) |
| **Computer Vision** | 1,011 (19%) | 1,021 (19%) |
| **Reinforcement Learning** | 991 (19%) | 802 (15%) |
| **Generative Models** | 812 (15%) | 736 (14%) |

**Key Observations:**
- ICLR 2026 has stronger LLM focus (40% vs 31%)
- NeurIPS 2025 more balanced across topics
- Similar computer vision coverage
- Both conferences show heavy ML systems focus

## Technical Stack

### Data Collection
- **API**: OpenReview API v2
- **Rate Limiting**: Conservative (3s retry, 2s page delay)
- **Authentication**: Required for better rate limits

### Analysis Pipeline
- **Clustering**: scikit-learn (KMeans, TF-IDF, SVD)
- **Topic Extraction**: Rule-based keyword matching
- **Storage**: Parquet (fast columnar format)

### UI
- **Framework**: Streamlit
- **Caching**: Automatic data caching
- **Similarity**: Cosine similarity on embeddings

## Project Structure

```
paper_crawler/
├── data/                    # ICLR 2026 data
├── analysis/                # ICLR 2026 analysis
├── app/                     # ICLR 2026 UI
├── neurips2025/            # NeurIPS 2025 system
│   ├── data/
│   ├── analysis/
│   └── app/
├── crawler/                 # Shared crawler code
│   ├── run_crawl.py
│   └── openreview_client.py
├── analysis/                # Shared analysis scripts
│   ├── cluster_analysis.py
│   ├── corpus_analysis.py
│   └── topic_extraction.py
└── README.md               # This file
```

## Adding New Conferences

To add a new conference (e.g., ICML 2025):

```bash
# 1. Create directory structure
mkdir -p icml2025/{data,analysis,app}

# 2. Crawl papers
python crawler/run_crawl.py \
    --conference ICML \
    --year 2025 \
    --out-dir icml2025/data

# 3. Run analysis
python analysis/cluster_analysis.py --input icml2025/data/papers.parquet --out-dir icml2025/analysis
python analysis/corpus_analysis.py --input icml2025/data/papers.parquet --out-dir icml2025/analysis
python analysis/topic_extraction.py --input icml2025/data/papers.parquet --out-dir icml2025/analysis

# 4. Copy and modify UI
cp app/streamlit_app.py icml2025/app/
# Edit paths in icml2025/app/streamlit_app.py

# 5. Launch on new port
streamlit run icml2025/app/streamlit_app.py --server.port 8503
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- openreview-py
- pandas
- numpy
- scikit-learn
- streamlit
- pyarrow

## Configuration

### OpenReview Credentials
Set environment variables or pass as arguments:
```bash
export OPENREVIEW_USERNAME="your_email@domain.com"
export OPENREVIEW_PASSWORD="your_password"
```

### Rate Limiting
Adjust in run_crawl.py:
- `--sleep-s`: Seconds between retries (default: 3)
- `--page-sleep`: Seconds between pages (default: 2)
- `--max-retries`: Max retry attempts (default: 5)

## Performance

| Operation | ICLR 2026 | NeurIPS 2025 |
|-----------|-----------|--------------|
| **Crawl** | ~57s | ~20s |
| **Clustering** | ~19s | ~6s |
| **Topic Extraction** | ~1s | ~1s |
| **Total** | ~80s | ~30s |

## Troubleshooting

### Rate Limit Errors
- Increase `--sleep-s` and `--page-sleep`
- Use authenticated requests
- Wait 15 seconds between runs

### UI Not Loading
- Check if port is already in use
- Try different port: `--server.port 8503`
- Clear Streamlit cache: `streamlit cache clear`

### Missing Papers
- Check venueid filter matches conference
- Verify invitation string is correct
- Some conferences use different field names

## Future Enhancements

- [ ] Author collaboration network
- [ ] Citation analysis
- [ ] Trend analysis across years
- [ ] Multi-conference comparison view
- [ ] Export to CSV/Excel
- [ ] PDF download integration
- [ ] Semantic search with embeddings

## License

MIT License - feel free to use and modify for your research needs.

## Credits

Built using OpenReview API and open-source tools.
