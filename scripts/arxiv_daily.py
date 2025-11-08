import os, time, datetime as dt
import feedparser
import pandas as pd
from urllib.parse import urlencode
from tenacity import retry, wait_fixed, stop_after_attempt

ARXIV_API = "https://export.arxiv.org/api/query"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def arxiv_query(params: dict) -> str:
    return f"{ARXIV_API}?{urlencode(params)}"

@retry(wait=wait_fixed(5), stop=stop_after_attempt(5))
def fetch(url: str):
    # arXiv ToU: single connection and AT LEAST 3s between requests
    time.sleep(3)
    # feedparser respects the URL and returns a parsed feed
    return feedparser.parse(url)

def iter_today_entries(max_results=2000, page_size=200, sort_by="submittedDate"):
    today = dt.datetime.utcnow().date()
    fetched, start = 0, 0
    while fetched < max_results:
        url = arxiv_query({
            "search_query": "all:*",
            "start": start,
            "max_results": page_size,
            "sortBy": sort_by,
            "sortOrder": "descending",
        })
        feed = fetch(url)
        entries = getattr(feed, "entries", [])
        if not entries:
            break
        for e in entries:
            pub_date = dt.datetime.fromisoformat(e.published.replace("Z","+00:00")).date()
            if pub_date < today:
                return
            yield e
        fetched += len(entries)
        start += page_size

def parse_entry(e):
    entry_id = e.id.split('/')[-1]               # e.g., 2501.01234v2
    base_id, version = entry_id.split('v')       # "2501.01234", "2"
    # links
    pdf_link = next((l.href for l in e.links if getattr(l, "type", "")=="application/pdf"), None)
    abs_link = next((l.href for l in e.links if getattr(l, "rel", "")=="alternate"), None)
    # categories
    cats = [t['term'] for t in getattr(e, "tags", [])]
    # authors
    authors = []
    for a in getattr(e, "authors", []):
        authors.append({
            "paper_id_version": entry_id,
            "author_name": a.name,
            "affiliation": getattr(a, "affiliation", None)
        })
    rec = {
        "paper_id_version": entry_id,
        "paper_id": base_id,
        "version": int(version),
        "title": e.title,
        "summary": e.summary,
        "published": e.published,
        "updated": getattr(e, "updated", e.published),
        "doi": getattr(e, "arxiv_doi", None),
        "journal_ref": getattr(e, "arxiv_journal_ref", None),
        "comment": getattr(e, "arxiv_comment", None),
        "primary_category": getattr(e, "arxiv_primary_category", {}).get("term")
            if hasattr(e, "arxiv_primary_category") else (cats[0] if cats else None),
        "all_categories": cats,
        "pdf_url": pdf_link,
        "abs_url": abs_link,
    }
    return rec, authors, [{"paper_id_version": entry_id, "category": c} for c in cats]

def load_parquet(path, cols):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=cols)

def main():
    papers_path = os.path.join(DATA_DIR, "papers.parquet")
    authors_path = os.path.join(DATA_DIR, "authors.parquet")
    cats_path = os.path.join(DATA_DIR, "categories.parquet")
    log_path = os.path.join(DATA_DIR, "run_log.csv")

    papers = load_parquet(papers_path, [
        "paper_id_version","paper_id","version","title","summary","published","updated",
        "doi","journal_ref","comment","primary_category","all_categories","pdf_url","abs_url"
    ])
    authors = load_parquet(authors_path, ["paper_id_version","author_name","affiliation"])
    categories = load_parquet(cats_path, ["paper_id_version","category"])

    new_papers, new_authors, new_cats = [], [], []

    for e in iter_today_entries():
        rec, a_list, c_list = parse_entry(e)
        new_papers.append(rec)
        new_authors.extend(a_list)
        new_cats.extend(c_list)

    if not new_papers:
        print("No new papers today (UTC).")
        return

    np_df = pd.DataFrame(new_papers).drop_duplicates("paper_id_version")
    na_df = pd.DataFrame(new_authors).drop_duplicates(["paper_id_version","author_name","affiliation"])
    nc_df = pd.DataFrame(new_cats).drop_duplicates(["paper_id_version","category"])

    papers = pd.concat([papers, np_df], ignore_index=True).drop_duplicates("paper_id_version", keep="last")
    authors = pd.concat([authors, na_df], ignore_index=True).drop_duplicates(["paper_id_version","author_name","affiliation"])
    categories = pd.concat([categories, nc_df], ignore_index=True).drop_duplicates(["paper_id_version","category"])

    papers.to_parquet(papers_path, index=False)
    authors.to_parquet(authors_path, index=False)
    categories.to_parquet(cats_path, index=False)

    run = pd.DataFrame([{
        "run_utc": pd.Timestamp.utcnow().isoformat(),
        "new_papers": len(np_df),
        "total_papers": len(papers)
    }])
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        logs = pd.concat([logs, run], ignore_index=True)
    else:
        logs = run
    logs.to_csv(log_path, index=False)

    print(f"Added {len(np_df)} papers; total now {len(papers)}.")

if __name__ == "__main__":
    main()
