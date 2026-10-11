# Tamil Nadu Agriculture Schemes Assistant (RAG)

A beginner-friendly RAG project using **Playwright, LangChain, OpenAI, FAISS, and Streamlit**.

## How it works

1. `ingest.py` uses Playwright to browse a short allow-list of official government sites.
2. It collects relevant page text and PDF links, downloads PDFs, and extracts text with PyMuPDF.
3. Text is split into chunks and embedded with OpenAI's `text-embedding-3-small`.
4. FAISS saves the vector index in `vectorstore/`.
5. `app.py` loads that saved index. For every question, `retriever.py` searches FAISS and sends only retrieved excerpts to the OpenAI chat model.
6. The UI displays the answer and source excerpts/links, including PDF page numbers where available.

**Important:** The Streamlit query path does not use Playwright, web search, or live page fetching. Internet access and an OpenAI API key are required during ingestion (to visit sites and create embeddings) and during answering (to create query embeddings and generate answers). The source text used to answer questions comes from the persisted index.

## 1. Requirements

- Windows 10/11
- Python 3.11 recommended
- An OpenAI API key with API access/billing enabled
- Internet access during ingestion and when asking questions
- Optional: Tesseract OCR if you want to extend the app to scanned PDFs

## 2. Setup in Windows PowerShell

Open PowerShell and move to the folder where you want the project:

```powershell
cd "C:\Learning\ProjectFiles\AI Master Class\Repo\assignments\Week2\Day5"
```

If you downloaded/extracted this project elsewhere, `cd` to that extracted folder instead.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Configure your OpenAI key securely

Copy the example environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `your_openai_api_key_here` with your actual API key. Do not add quotes unless needed, and never paste the real key into Python source code or commit `.env` to GitHub.

Confirm `.env` is ignored by Git:

```powershell
git check-ignore .env
```

If a real key was already committed, remove it from the repository and revoke/rotate the key in the OpenAI dashboard. Adding `.env` to `.gitignore` does not remove a key from previous commits.

## 4. Build the knowledge base

```powershell
python ingest.py
```

This may take time and uses API credits for embeddings. Check `logs/ingest.log` if a website or PDF fails. You can also manually place official scheme PDFs in `data/pdfs/` and rerun ingestion.

The starter crawler has an allow-list and a page limit. If official department websites change their structure, update `START_URLS` and `ALLOWED_DOMAINS` in `config.py`. Verify the current scheme details and the officialness of source pages before using the app for real decisions.

## 5. Start Streamlit

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

You only need to run `ingest.py` when adding or refreshing source documents. Do not run ingestion for every question.

## Project layout

```text
tamilnadu-agriculture-rag/
├── app.py
├── config.py
├── ingest.py
├── retriever.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── pdfs/
│   ├── extracted/
│   ├── media/
│   └── manifest.json       # created by ingestion
├── logs/
└── vectorstore/            # persisted FAISS files created by ingestion
```

## Scope and limitations

- **PDF extraction:** text-based PDF pages and page numbers are supported. This starter version does not perform OCR automatically, so scanned/image-only pages may produce no text. Install Tesseract and add OCR processing to support those PDFs.
- **Tables:** PyMuPDF extracts page text; complex table layouts may not retain exact row/column structure.
- **Images/audio/video:** this starter does not claim to understand embedded images, audio, or video. It records/extracts readable webpage text and PDF text only. Add OCR, image analysis, transcription, or video-frame extraction as separate pipelines if needed.
- **Linked URLs:** the crawler follows relevant links only within the configured allow-list, with a per-domain cap. It does not crawl arbitrary external URLs.
- **Tamil:** retrieval quality depends on the selected embedding model and the language/content quality of the source documents. Test it with real Tamil questions.
- **Similarity threshold:** FAISS distances depend on the embedding/index configuration. `MAX_DISTANCE` is a tunable starting point, not a universal confidence score. Calibrate it against known answerable and unanswerable questions.
- **Knowledge freshness:** answers only know what has been indexed. Run ingestion again to refresh official content.
- **API costs:** OpenAI chat and embedding API calls may incur charges; API usage is not necessarily included in a ChatGPT subscription.

## Troubleshooting

- `OPENAI_API_KEY is missing`: check `.env` filename, key name, and that you launched the app from this project.
- Missing index: run `python ingest.py` successfully first.
- Empty index: inspect `logs/ingest.log`, verify the official URLs, and manually add official PDFs to `data/pdfs/`.
- API quota/rate limit: check OpenAI API billing/usage and retry later.
- Unsupported scanned PDF: add OCR or obtain a text-based official PDF.
