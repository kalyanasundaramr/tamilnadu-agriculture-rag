from ingest import valid_official_url

def test_only_allowlisted_https_domain_is_allowed():
    assert valid_official_url("https://www.tnagriculture.in/")
    assert not valid_official_url("https://example.com/")
    assert not valid_official_url("file:///tmp/document.pdf")
