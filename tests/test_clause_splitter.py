from src.clause_splitter import extract_clauses


def test_abbreviations_do_not_split():
    out = extract_clauses("The Company shall indemnify the Client, e.g. for losses caused by Acme Inc. in full.")
    assert len(out) == 1


def test_numbered_headings_are_stripped_and_split():
    text = "1. Payment shall be made within thirty days of invoice.\n2. Either party may terminate on ninety days notice."
    out = extract_clauses(text)
    assert out == ["Payment shall be made within thirty days of invoice",
                   "Either party may terminate on ninety days notice"]


def test_soft_line_wraps_are_joined():
    out = extract_clauses("The Tenant shall pay rent on or\nbefore the fifth day of each month.")
    assert len(out) == 1 and "\n" not in out[0]


def test_real_clauses_with_date_or_name_words_are_kept():
    out = extract_clauses("Payment shall be made on the due date stated in each invoice.")
    assert len(out) == 1


def test_signature_noise_is_dropped():
    assert extract_clauses("Signature: ______________________") == []
    assert extract_clauses("Page 3 of 10") == []


def test_empty_input():
    assert extract_clauses("") == [] and extract_clauses("   ") == []
