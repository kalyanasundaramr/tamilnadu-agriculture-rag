"""Download official scheme PDFs/pages, extract text, and build a persistent FAISS index.

Run once to create/update the knowledge base: python ingest.py
The app never calls this script automatically.
"""
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag

import pymupdf
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from config import (
    ALLOWED_DOMAINS, CHAT_MODEL, EMBEDDING_MODEL, LOG_PATH, MANIFEST_PATH,
    MAX_PAGES_PER_DOMAIN, PDF_DIR, EXTRACTED_DIR, START_URLS, CHUNK_OVERLAP,
    CHUNK_SIZE, ensure_directories, OPENAI_API_KEY, REQUEST_TIMEOUT_MS, VECTOR_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("ingest")
HEADERS = {"User-Agent": "TamilNaduAgricultureSchemesResearchBot/1.0 (educational project)"}

def valid_official_url(url):
    p = urlparse(url)
    return p.scheme in ("http", "https") and p.hostname in ALLOWED_DOMAINS

def safe_filename(value, suffix):
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:90].strip("._") or "document"
    return f"{base}{suffix}"

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Manifest was unreadable; starting a fresh manifest.")
    return {"documents": []}

def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def extract_pdf(path, source_url):
    docs = []
    try:
        with pymupdf.open(path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": source_url,
                            "filename": path.name,
                            "title": path.stem,
                            "page": page_no,
                            "content_type": "pdf",
                        },
                    ))
                # Capture clickable links for audit/manifest; link targets are selectively crawled separately.
    except Exception as exc:
        log.exception("PDF extraction failed for %s: %s", path, exc)
    return docs

def download_pdf(url, session, manifest):
    try:
        response = session.get(url, timeout=35, headers=HEADERS)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            return None
        digest = sha256_bytes(response.content)
        existing = next((x for x in manifest["documents"] if x.get("sha256") == digest), None)
        if existing:
            log.info("Duplicate PDF skipped: %s", url)
            return existing.get("filename")
        filename = safe_filename(urlparse(url).path.split("/")[-1] or "scheme.pdf", ".pdf")
        if not filename.lower().endswith(".pdf"):
            filename = filename.rsplit(".", 1)[0] + ".pdf"
        # Keep same-name files distinct by content hash.
        filename = f"{PathStem(filename)}_{digest[:8]}.pdf"
        target = PDF_DIR / filename
        target.write_bytes(response.content)
        manifest["documents"].append({
            "title": filename, "url": url, "filename": filename, "sha256": digest,
            "downloaded_at": datetime.now(timezone.utc).isoformat(), "status": "downloaded",
            "kind": "pdf",
        })
        log.info("Downloaded PDF: %s", url)
        return filename
    except Exception as exc:
        log.warning("PDF download failed %s: %s", url, exc)
        manifest["documents"].append({"url": url, "status": "failed", "error": str(exc), "kind": "pdf"})
        return None

def PathStem(filename):
    return filename[:-4] if filename.lower().endswith(".pdf") else filename

def crawl_official_sites():
    """Use Playwright to collect page text and PDF links from allow-listed sites only."""
    session = requests.Session()
    manifest = load_manifest()
    page_docs = []
    seen_urls = set()
    pdf_urls = set()
    pages_per_domain = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a context so certificate handling is explicit and each URL gets its own page.
        # This is a workaround for misconfigured certificates on listed source sites; it does
        # not validate the site's identity. Only crawl the allow-listed government sources.
        context = browser.new_context(ignore_https_errors=True)
        context.set_default_timeout(REQUEST_TIMEOUT_MS)
        queue = list(START_URLS)
        try:
            while queue:
                url = urldefrag(queue.pop(0))[0]
                if url in seen_urls or not valid_official_url(url):
                    continue
                domain = urlparse(url).hostname
                if pages_per_domain.get(domain, 0) >= MAX_PAGES_PER_DOMAIN:
                    continue
                seen_urls.add(url)
                pages_per_domain[domain] = pages_per_domain.get(domain, 0) + 1
                page = context.new_page()
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
                    if response is not None and response.status >= 400:
                        log.warning("HTTP %s while opening %s", response.status, url)
                        continue
                    page.wait_for_timeout(300)
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")
                    title = soup.title.get_text(" ", strip=True) if soup.title else url
                    for a in soup.find_all("a", href=True):
                        target = urljoin(url, a["href"])
                        if not valid_official_url(target):
                            continue
                        parsed_target = urlparse(target)
                        if parsed_target.path.lower().endswith(".pdf") or ".pdf?" in target.lower():
                            pdf_urls.add(target)
                        elif target not in seen_urls and len(queue) < 300:
                            label = a.get_text(" ", strip=True).lower()
                            if any(word in (label + " " + target).lower() for word in
                                   ("agri", "scheme", "farmer", "subsid", "horti", "insurance",
                                    "irrigation", "department", "benefit", "pm-kisan", "crop")):
                                queue.append(target)
                    text = soup.get_text("\n", strip=True)
                    if len(text) > 80:
                        page_docs.append(Document(
                            page_content=text,
                            metadata={"source": url, "title": title, "filename": "", "page": None,
                                      "content_type": "webpage"},
                        ))
                        (EXTRACTED_DIR / f"{sha256_bytes(url.encode())[:12]}.txt").write_text(
                            f"TITLE: {title}\nURL: {url}\n\n{text}", encoding="utf-8"
                        )
                    log.info("Crawled %s (text chars: %s)", url, len(text))
                    time.sleep(0.4)
                except PlaywrightTimeoutError:
                    log.warning("Page timed out: %s", url)
                except Exception as exc:
                    log.warning("Page crawl failed %s: %s", url, exc)
                finally:
                    page.close()
        finally:
            context.close()
            browser.close()

    log.info("Crawl discovery summary: %s webpage documents, %s PDF links found", len(page_docs), len(pdf_urls))
    pdf_docs = []
    for url in sorted(pdf_urls):
        filename = download_pdf(url, session, manifest)
        if filename:
            pdf_docs.extend(extract_pdf(PDF_DIR / filename, url))

    # Save page entries to manifest, de-duplicating by URL.
    known = {x.get("url") for x in manifest["documents"]}
    for doc in page_docs:
        if doc.metadata["source"] not in known:
            manifest["documents"].append({
                "title": doc.metadata["title"], "url": doc.metadata["source"],
                "status": "processed", "kind": "webpage",
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
            known.add(doc.metadata["source"])
    save_manifest(manifest)
    log.info("Extraction summary: %s webpage records, %s PDF page records", len(page_docs), len(pdf_docs))
    return page_docs + pdf_docs, manifest

def build_index(documents):
    if not documents:
        raise RuntimeError(
            "No readable content was collected. Check internet access, official site URLs, "
            "logs/ingest.log, and whether the sites expose text/PDF links."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Add a stable content hash so re-running ingestion doesn't add duplicate chunks.
    unique = {}
    for doc in chunks:
        key = sha256_bytes((doc.metadata.get("source", "") + "|" +
                            str(doc.metadata.get("page", "")) + "|" + doc.page_content).encode("utf-8"))
        doc.metadata["chunk_id"] = key
        unique[key] = doc
    chunks = list(unique.values())
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(VECTOR_DIR), index_name="faiss_index")
    log.info("Saved %s unique chunks to %s", len(chunks), VECTOR_DIR)
    return len(chunks)

def main():
    ensure_directories()
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is missing. Add it to .env; see .env.example.")
    log.info("Starting official-source ingestion.")
    documents, manifest = crawl_official_sites()
    # Also support PDFs manually placed in data/pdfs, including documents obtained from official sources.
    already = {d.metadata.get("filename") for d in documents if d.metadata.get("filename")}
    for pdf_path in PDF_DIR.glob("*.pdf"):
        if pdf_path.name not in already:
            documents.extend(extract_pdf(pdf_path, f"file://{pdf_path.resolve()}"))
    count = build_index(documents)
    manifest["last_ingestion_at"] = datetime.now(timezone.utc).isoformat()
    manifest["indexed_chunk_count"] = count
    manifest["processed_document_count"] = len(documents)
    save_manifest(manifest)
    print("\nIngestion complete")
    print(f"Source records: {len(manifest['documents'])}")
    print(f"Extracted document/page records: {len(documents)}")
    print(f"Indexed chunks: {count}")
    print(f"FAISS directory: {VECTOR_DIR}")
    print(f"Log: {LOG_PATH}")

if __name__ == "__main__":
    main()
