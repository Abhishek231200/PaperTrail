"""Two chunking strategies, both operating on (pmid, title, abstract).

whole_abstract: one chunk per paper (title + full abstract).
sentence_window: overlapping windows of N sentences, each prefixed with the
title so a chunk read in isolation still carries paper-level context.
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences


def whole_abstract(title: str, abstract: str) -> list[str]:
    return [f"{title}\n\n{abstract}"]


def sentence_window(title: str, abstract: str, window: int = 3, overlap: int = 1) -> list[str]:
    sentences = split_sentences(abstract)
    if not sentences:
        return [f"{title}\n\n{abstract}"]
    step = max(window - overlap, 1)
    chunks = []
    for start in range(0, len(sentences), step):
        window_sentences = sentences[start:start + window]
        if not window_sentences:
            continue
        chunks.append(f"{title}\n\n" + " ".join(window_sentences))
        if start + window >= len(sentences):
            break
    return chunks


def chunk_paper(strategy: str, title: str, abstract: str, window: int = 3, overlap: int = 1) -> list[str]:
    if strategy == "whole_abstract":
        return whole_abstract(title, abstract)
    if strategy == "sentence_window":
        return sentence_window(title, abstract, window, overlap)
    raise ValueError(f"unknown chunking strategy: {strategy}")
