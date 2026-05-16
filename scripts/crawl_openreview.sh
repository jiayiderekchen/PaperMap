#!/bin/bash
# Generic OpenReview crawler pipeline (NeurIPS, ICLR, ICML)
#
# Usage:
#   CONFERENCE=NeurIPS YEAR=2025 bash scripts/crawl_openreview.sh
#
# Required env vars:
#   CONFERENCE   — NeurIPS | ICLR | ICML
#   YEAR         — e.g. 2025
#
# Optional env vars:
#   OPENREVIEW_USERNAME  — your OpenReview email (public conferences work without login)
#   OPENREVIEW_PASSWORD  — your OpenReview password

set -e

CONFERENCE=${CONFERENCE:?'Set CONFERENCE env var (NeurIPS | ICLR | ICML)'}
YEAR=${YEAR:?'Set YEAR env var (e.g. 2025)'}
CONF_LOWER=$(echo "$CONFERENCE" | tr '[:upper:]' '[:lower:]')
OUT_DIR="${CONF_LOWER}${YEAR}/data"
ANALYSIS_DIR="${CONF_LOWER}${YEAR}/analysis"

echo "=== $CONFERENCE $YEAR Paper Crawler ==="
echo ""

source .venv/bin/activate

echo "Step 1: Crawling $CONFERENCE $YEAR accepted papers..."
python crawler/run_crawl.py \
    ${OPENREVIEW_USERNAME:+--username "$OPENREVIEW_USERNAME"} \
    ${OPENREVIEW_PASSWORD:+--password "$OPENREVIEW_PASSWORD"} \
    --conference "$CONFERENCE" \
    --year "$YEAR" \
    --out-dir "$OUT_DIR" \
    --sleep-s 3 \
    --page-sleep 2 \
    --limit 5000 \
    --max-retries 5 \
    --base-url "https://api2.openreview.net"

if [ ! -s "$OUT_DIR/papers.parquet" ]; then
    echo "Warning: No papers crawled. Data may not be publicly released yet."
    exit 0
fi

echo ""
echo "Step 2: Cluster analysis..."
python analysis/cluster_analysis.py \
    --input "$OUT_DIR/papers.parquet" \
    --out-dir "$ANALYSIS_DIR" \
    --clusters 50 --svd-dims 100 --max-features 5000

echo ""
echo "Step 3: Corpus analysis..."
python analysis/corpus_analysis.py \
    --input "$OUT_DIR/papers.parquet" \
    --out-dir "$ANALYSIS_DIR"

echo ""
echo "Step 4: Topic extraction..."
python analysis/topic_extraction.py \
    --input "$OUT_DIR/papers.parquet" \
    --out-dir "$ANALYSIS_DIR"

echo ""
echo "=== Pipeline Complete ==="
echo "Papers  : $OUT_DIR/papers.parquet"
echo "Analysis: $ANALYSIS_DIR/"
echo ""
echo "Launch UI:"
echo "  streamlit run app/streamlit_app.py --server.port 8501"
