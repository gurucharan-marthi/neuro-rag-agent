"""Retrieval functions: local Qdrant search + PubMed tool for the agent."""
import json
import xml.etree.ElementTree as ET

import requests
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DB_PATH = "data/qdrant"
COLLECTION = "papers"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Load once, reuse for every query
_model = SentenceTransformer(EMBED_MODEL)
_client = QdrantClient(path=DB_PATH)


def search_local(query: str, k: int = 5) -> list[dict]:
    """Search the local arXiv index."""
    vec = _model.encode(query, normalize_embeddings=True).tolist()
    hits = _client.query_points(COLLECTION, query=vec, limit=k).points
    return [
        {"title": h.payload["title"], "abstract": h.payload["abstract"],
         "url": h.payload["url"], "source": "arxiv"}
        for h in hits
    ]


@tool
def pubmed_search(query: str, max_results: int = 5) -> str:
    """Search PubMed for biomedical and neuroimaging papers.
    Use a short keyword query, e.g. 'optic chiasm MRI segmentation'.
    Returns a JSON list of papers with title, abstract and url."""
    ids = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results,
                "retmode": "json", "tool": "neuro-rag-agent"},
        timeout=20,
    ).json()["esearchresult"]["idlist"]
    if not ids:
        return "[]"

    xml = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=20,
    ).text

    papers = []
    for art in ET.fromstring(xml).findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
        papers.append({"title": title, "abstract": abstract,
                       "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                       "source": "pubmed"})
    return json.dumps(papers)
