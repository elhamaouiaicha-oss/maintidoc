"""Modèles d'embedding open source (chargés à la demande).

- Texte  : BGE-M3 (multilingue FR/EN), via sentence-transformers.
- Images : CLIP (clip-ViT-B-32), espace commun texte <-> image.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .config import settings


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.text_embed_model)
    model.max_seq_length = 512  # suffisant pour nos passages, limite la RAM
    return model


@lru_cache(maxsize=1)
def _clip_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.clip_model)


def embed_texts(texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
    """Embeddings normalisés (distance cosinus) pour des passages ou requêtes."""
    if not texts:
        return []
    vecs = _text_model().encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def clip_embed_images(paths: Sequence[str | Path]) -> list[list[float]]:
    from PIL import Image

    if not paths:
        return []
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
    vecs = _clip_model().encode(
        images, batch_size=8, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


def clip_embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Encode des requêtes dans l'espace CLIP (le texte est tronqué à 77 tokens)."""
    if not texts:
        return []
    vecs = _clip_model().encode(
        list(texts), normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()
