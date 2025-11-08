import os
import pandas as pd

ENRICH_PATH = "data/enrichments.parquet"
PAPERS_PATH = "data/papers.parquet"

def simple_keywords(text: str):
    text = (text or "").lower()
    tags = []
    if any(k in text for k in ["large language model", "llm", "gpt"]): tags.append("llm")
    if "reinforcement learning" in text or "rl " in text: tags.append("reinforcement-learning")
    if "diffusion" in text: tags.append("diffusion")
    if "graph neural" in text or "gnn" in text: tags.append("gnn")
    if "federated" in text: tags.append("federated-learning")
    return list(set(tags))

def main():
    if not os.path.exists(PAPERS_PATH):
        print("No papers yet.")
        return
    df = pd.read_parquet(PAPERS_PATH)
    today = pd.Timestamp.utcnow().date()
    today_rows = df[pd.to_datetime(df["published"]).dt.date == today].copy()
    if today_rows.empty:
        print("Nothing to enrich today.")
        return

    rows = []
    for _, r in today_rows.iterrows():
        text = f"{r.get('title','')}\n\n{r.get('summary','')}"
        tags = simple_keywords(text)
        rows.append({
            "paper_id_version": r["paper_id_version"],
            "tags": tags,
            "has_code": any(k in text.lower() for k in ["github.com", "code", "implementation"])
        })

    enr = pd.DataFrame(rows)
    if os.path.exists(ENRICH_PATH):
        prev = pd.read_parquet(ENRICH_PATH)
        enr = pd.concat([prev, enr], ignore_index=True).drop_duplicates("paper_id_version", keep="last")
    enr.to_parquet(ENRICH_PATH, index=False)
    print(f"Enriched {len(rows)} items.")

if __name__ == "__main__":
    main()
