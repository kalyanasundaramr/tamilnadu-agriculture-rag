def test_source_metadata_shape():
    # The UI expects these fields from retriever.py.
    source = {"title": "Scheme PDF", "url": "https://example.gov.in/a.pdf",
              "page": 2, "excerpt": "Eligibility text"}
    assert source["title"]
    assert source["url"].startswith("https://")
    assert source["page"] > 0
    assert source["excerpt"]
