from retriever import clean_source_title


def test_strips_pdf_extension_and_hash():
    raw = "Revamped_Operational_Guidelines_17th_August_2020.pdf_fc57c66e"
    assert clean_source_title(raw) == "Revamped Operational Guidelines 17th August 2020"


def test_strips_hash_before_extension():
    assert clean_source_title("Crop_Insurance_Scheme_1a2b3c4d.pdf") == "Crop Insurance Scheme"


def test_keeps_webpage_titles():
    title = "Department of Agriculture - Government of Tamil Nadu"
    assert clean_source_title(title) == title


def test_empty_title_fallback():
    assert clean_source_title("") == "Official source"
    assert clean_source_title(None) == "Official source"
