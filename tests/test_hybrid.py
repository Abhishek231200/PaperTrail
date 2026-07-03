from src.retrieval.hybrid import rrf_fuse


def _item(pmid, chunk_ix=0, text="text"):
    return {"pmid": pmid, "chunk_ix": chunk_ix, "text": text}


def test_rrf_fuse_agreement_boosts_rank():
    dense = [_item(1), _item(2), _item(3)]
    lexical = [_item(2), _item(1), _item(4)]
    fused = rrf_fuse([dense, lexical], rrf_k=60)
    fused_pmids = [f["pmid"] for f in fused]
    # pmid 1 and 2 appear in both lists (near the top of each) so should
    # outrank pmid 3 and 4, which each appear in only one list.
    assert fused_pmids[0] in (1, 2)
    assert fused_pmids[1] in (1, 2)
    assert set(fused_pmids[2:]) == {3, 4}


def test_rrf_fuse_dedupes_by_pmid_and_chunk_ix():
    dense = [_item(1, chunk_ix=0), _item(1, chunk_ix=1)]
    fused = rrf_fuse([dense], rrf_k=60)
    # different chunk_ix for the same pmid are distinct retrieval units
    assert len(fused) == 2


def test_rrf_fuse_empty_lists():
    assert rrf_fuse([[], []], rrf_k=60) == []


def test_rrf_fuse_single_list_preserves_order():
    items = [_item(5), _item(6), _item(7)]
    fused = rrf_fuse([items], rrf_k=60)
    assert [f["pmid"] for f in fused] == [5, 6, 7]
