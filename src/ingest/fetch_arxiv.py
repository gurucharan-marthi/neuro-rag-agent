"""Download arXiv paper metadata + abstracts for neuroimaging topics."""
import json
import arxiv

QUERIES = [
    "diffusion MRI tractography",
    "white matter tractography deep learning",
    "brain MRI segmentation deep learning",
    "medical image registration deep learning",
    "optic nerve MRI",
    "visual pathway MRI",
    "diffusion tensor imaging",
    "neuroimaging foundation model",
]
PER_QUERY = 300
OUT = "data/raw/arxiv.jsonl"

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
seen = set()

with open(OUT, "w") as f:
    for q in QUERIES:
        search = arxiv.Search(query=q, max_results=PER_QUERY,
                              sort_by=arxiv.SortCriterion.Relevance)
        n = 0
        for r in client.results(search):
            pid = r.get_short_id()
            if pid in seen:
                continue
            seen.add(pid)
            f.write(json.dumps({
                "id": pid,
                "title": r.title,
                "abstract": r.summary.replace("\n", " "),
                "authors": [a.name for a in r.authors],
                "published": r.published.strftime("%Y-%m-%d"),
                "categories": r.categories,
                "url": r.entry_id,
                "query": q,
            }) + "\n")
            n += 1
        print(f"{q}: {n} new papers")

print(f"Total: {len(seen)} papers -> {OUT}")
