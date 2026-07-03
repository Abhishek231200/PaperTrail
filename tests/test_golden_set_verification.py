from src.evals.build_golden_set import quote_supported


def test_quote_supported_exact_match():
    assert quote_supported("tirzepatide reduced weight", "Patients on tirzepatide reduced weight over time.")


def test_quote_supported_normalizes_whitespace():
    quote = "tirzepatide   reduced\nweight"
    source = "Patients on tirzepatide reduced weight over time."
    assert quote_supported(quote, source)


def test_quote_supported_case_insensitive():
    assert quote_supported("TIRZEPATIDE reduced weight", "tirzepatide reduced weight over time")


def test_quote_supported_rejects_paraphrase():
    assert not quote_supported("weight went down a lot", "Patients on tirzepatide reduced weight over time.")


def test_quote_supported_rejects_fabricated_number():
    source = "87.5% of participants achieved the threshold."
    assert not quote_supported("92.1% of participants achieved the threshold.", source)
