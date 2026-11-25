"""Tests de l'extraction et de la fusion RRF (sans base ni modèles)."""

from pathlib import Path

import pytest

from maintidoc.extraction import extract_pdf, rows_to_markdown
from maintidoc.retrieval import rrf_fuse

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "attention_is_all_you_need.pdf"


def test_rows_to_markdown_basic():
    md = rows_to_markdown([["A", "B"], ["1", None], ["", ""]])
    assert md.splitlines()[0] == "| A | B |"
    assert "| 1 |  |" in md


def test_rows_to_markdown_rejects_degenerate_tables():
    assert rows_to_markdown([["seule ligne", "x"]]) == ""
    assert rows_to_markdown([["a", ""], ["b", ""]]) == ""  # une seule colonne utile


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"}
    assert fused[-1] in {"c", "d"}


@pytest.mark.skipif(not SAMPLE.exists(), reason="PDF d'exemple absent")
def test_extract_attention_paper(tmp_path):
    ex = extract_pdf(SAMPLE, images_root=tmp_path)
    assert ex.n_pages == 11
    assert {p for p, _ in ex.text_chunks} >= {1, 3, 4, 9}
    assert any("multi-head" in t.lower() for _, t in ex.text_chunks)

    captions = " ".join(t.caption for t in ex.tables)
    assert len(ex.tables) >= 3
    assert "Table 1" in captions and "Table 2" in captions and "Table 3" in captions
    table2 = next(t for t in ex.tables if t.page == 8)
    assert "23.75" in table2.markdown  # nombres décimaux non éclatés

    assert len(ex.images) >= 3
    assert any("Figure 1" in i.caption for i in ex.images)
    assert all(i.path.exists() and i.path.stat().st_size > 0 for i in ex.images)
