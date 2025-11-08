# Daily arXiv Metadata → Kaggle

Automated daily pipeline that:
1) Pulls arXiv papers submitted today (UTC)
2) Merges into rolling Parquet history
3) (Optional) adds lightweight tags
4) Publishes to Kaggle Datasets

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/arxiv_daily.py
python scripts/enrich_today.py  # optional
