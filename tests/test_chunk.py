from src.retrieval.chunk import chunk_paper, sentence_window, split_sentences, whole_abstract


def test_whole_abstract_returns_single_chunk():
    chunks = whole_abstract("Title", "Sentence one. Sentence two.")
    assert len(chunks) == 1
    assert "Title" in chunks[0]
    assert "Sentence one." in chunks[0]


def test_split_sentences_basic():
    sentences = split_sentences("First sentence. Second sentence. Third one!")
    assert sentences == ["First sentence.", "Second sentence.", "Third one!"]


def test_split_sentences_empty_string():
    assert split_sentences("") == []


def test_sentence_window_overlap():
    abstract = "One. Two. Three. Four. Five."
    chunks = sentence_window("T", abstract, window=3, overlap=1)
    # step = window - overlap = 2; loop stops once a window reaches the last
    # sentence, so windows start at indices 0 and 2 (not 4 -- that would be
    # a redundant single-sentence tail window already covered by chunk 1).
    assert len(chunks) == 2
    assert "One. Two. Three." in chunks[0]
    assert "Three. Four. Five." in chunks[1]


def test_sentence_window_prefixes_title_for_standalone_context():
    chunks = sentence_window("My Title", "One. Two. Three.", window=3, overlap=1)
    assert all(c.startswith("My Title") for c in chunks)


def test_sentence_window_falls_back_to_whole_abstract_when_no_sentences():
    chunks = sentence_window("Title", "", window=3, overlap=1)
    assert chunks == ["Title\n\n"]


def test_chunk_paper_dispatches_by_strategy():
    assert len(chunk_paper("whole_abstract", "T", "A.")) == 1
    assert chunk_paper("sentence_window", "T", "One. Two. Three.", window=3, overlap=1)


def test_chunk_paper_unknown_strategy_raises():
    try:
        chunk_paper("nonexistent", "T", "A.")
        assert False, "expected ValueError"
    except ValueError:
        pass
