"""Recherche vectorielle multimodale.

Deux espaces d'embedding distincts :
- BGE-M3 : passages, tableaux et descriptions d'images (recherche texte -> texte)
- CLIP   : images (recherche texte -> image)

`retrieve` fusionne les deux listes avec Reciprocal Rank Fusion (RRF), car les
scores cosinus de deux modèles différents ne sont pas comparables entre eux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

from . import db, models

RRF_K = 60


@dataclass
class Hit:
    kind: str  # "text" | "table" | "image"
    page: int | None
    content: str
    score: float
    doc_id: str
    chunk_id: int | None = None
    image_id: int | None = None
    image_path: str | None = None
    caption: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Hashable:
        if self.kind == "image":
            return ("img", self.image_id)
        return ("chunk", self.chunk_id)

    def describe(self) -> str:
        label = {"text": "texte", "table": "tableau", "image": "image"}[self.kind]
        page = f"p.{self.page}" if self.page else "p.?"
        file = self.metadata.get("filename", "")
        return f"{file} {page}, {label}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "page": self.page, "content": self.content,
            "score": round(float(self.score), 4), "doc_id": self.doc_id,
            "image_path": self.image_path, "caption": self.caption,
            "filename": self.metadata.get("filename", ""),
        }


def rrf_fuse(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[Hashable]:
    """Reciprocal Rank Fusion : renvoie les clés triées par score fusionné."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


# ------------------------------------------------------------ conversions
def _chunk_to_hit(row: dict[str, Any]) -> Hit:
    meta = row.get("metadata") or {}
    if row["kind"] == "image_description":
        return Hit(
            kind="image", page=row["page"], content=row["content"],
            score=float(row["score"]), doc_id=row["doc_id"],
            image_id=meta.get("image_id"), image_path=meta.get("path"),
            caption=meta.get("caption", ""), metadata=meta,
        )
    return Hit(
        kind=row["kind"], page=row["page"], content=row["content"],
        score=float(row["score"]), doc_id=row["doc_id"],
        chunk_id=row["id"], metadata=meta,
    )


def _image_to_hit(row: dict[str, Any]) -> Hit:
    meta = row.get("metadata") or {}
    content = row.get("description") or row.get("caption") or "(image sans description)"
    return Hit(
        kind="image", page=row["page"], content=content, score=float(row["score"]),
        doc_id=row["doc_id"], image_id=row["id"], image_path=row["path"],
        caption=row.get("caption") or "", metadata=meta,
    )


# ---------------------------------------------------------------- recherche
def search_passages(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    kinds: Sequence[str] = ("text",),
) -> list[Hit]:
    """Recherche texte -> texte (BGE-M3) restreinte aux types demandés."""
    qv = models.embed_query(query)
    with db.connect() as conn:
        rows = db.search_chunks(conn, qv, k=k, doc_id=doc_id, kinds=kinds)
    return [_chunk_to_hit(r) for r in rows]


def search_images(query: str, k: int = 3, doc_id: str | None = None) -> list[Hit]:
    """Recherche texte -> image : CLIP + descriptions indexées, fusionnés (RRF)."""
    clip_vec = models.clip_embed_texts([query])[0]
    qv = models.embed_query(query)
    with db.connect() as conn:
        clip_rows = db.search_images(conn, clip_vec, k=k, doc_id=doc_id)
        desc_rows = db.search_chunks(
            conn, qv, k=k, doc_id=doc_id, kinds=("image_description",)
        )
        by_key: dict[Hashable, Hit] = {}
        clip_rank, desc_rank = [], []
        for r in clip_rows:  # CLIP en premier : porte le chemin du fichier
            h = _image_to_hit(r)
            by_key[h.key] = h
            clip_rank.append(h.key)
        for r in desc_rows:
            h = _chunk_to_hit(r)
            desc_rank.append(h.key)
            if h.key not in by_key:
                img = db.get_image(conn, h.image_id) if h.image_id else None
                by_key[h.key] = _image_to_hit({**img, "score": r["score"]}) if img else h
    ordered = rrf_fuse([clip_rank, desc_rank])
    return [by_key[key] for key in ordered[:k]]


def retrieve(
    query: str,
    k_text: int = 5,
    k_images: int = 2,
    doc_id: str | None = None,
) -> list[Hit]:
    """Contexte multimodal : passages + tableaux + images, dédoublonnés."""
    text_hits = search_passages(query, k=k_text, doc_id=doc_id, kinds=("text", "table"))
    image_hits = search_images(query, k=k_images, doc_id=doc_id)
    seen, merged = set(), []
    for h in text_hits + image_hits:
        if h.key not in seen:
            seen.add(h.key)
            merged.append(h)
    return merged


def get_page(page: int, doc_id: str | None = None) -> list[Hit]:
    with db.connect() as conn:
        rows = db.get_page_chunks(conn, doc_id, page)
    return [_chunk_to_hit({**r, "score": 1.0}) for r in rows]
