#!/bin/bash
# AAAI 2026 Paper Crawler Pipeline
# Source: AAAI OJS proceedings at https://ojs.aaai.org/index.php/AAAI/issue/view/683
# AAAI 2026 (Vol. 40) spans 47 issues (issue IDs 683-729), one track per issue.

set -e

echo "=== AAAI 2026 Paper Crawler ==="
echo ""

# Activate virtual environment
source .venv/bin/activate

YEAR=2026
OUT_DIR="aaai2026/data"
ANALYSIS_DIR="aaai2026/analysis"

echo "Step 1: Scraping AAAI 2026 papers from ojs.aaai.org..."
echo "  (47 issues x ~100 papers each ≈ ~4500 papers, allow ~20 min)"
python crawler/crawl_aaai_ojs.py \
    --year $YEAR \
    --out-dir "$OUT_DIR" \
    --sleep-s 0.5 \
    --max-retries 4

PAPER_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('$OUT_DIR/papers.parquet'); print(len(df))")
echo "Crawled $PAPER_COUNT papers."

if [ "$PAPER_COUNT" -eq 0 ]; then
    echo "Error: No papers crawled. Check network access and try again."
    exit 1
fi

echo ""
echo "Step 2: Running cluster analysis..."
python analysis/cluster_analysis.py \
    --input "$OUT_DIR/papers.parquet" \
    --out-dir "$ANALYSIS_DIR" \
    --clusters 50 \
    --svd-dims 100 \
    --max-features 5000

echo ""
echo "Step 3: Running corpus analysis..."
python analysis/corpus_analysis.py \
    --input "$OUT_DIR/papers.parquet" \
    --out-dir "$ANALYSIS_DIR"

echo ""
echo "Step 4: Extracting topics..."
python analysis/topic_extraction.py \
    --input "$OUT_DIR/papers.parquet" \
    --out-dir "$ANALYSIS_DIR"

echo ""
echo "=== Pipeline Complete ==="
echo "Papers: $OUT_DIR/papers.parquet"
echo "Analysis: $ANALYSIS_DIR/"
echo ""
echo "To launch unified UI:"
echo "  streamlit run app/streamlit_app.py --server.port 8501"
