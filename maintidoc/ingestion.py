r"""Pipeline d'indexation multimodale : PDF -> embeddings -> PGVector.

    PDF --LlamaIndex--> passages texte ----------------\
        --PyMuPDF-----> tableaux (Markdown) -----------+--> BGE-M3 --> chunks
        --PyMuPDF-----> images --> VLM (description) --+--> BGE-M3 --> chunks
                               \-> CLIP (image) ----------------------> images
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from . import db, models
from .config import settings
from .extraction import ExtractedPDF, extract_pdf
from .vision import describe_image

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


def _noop(msg: str, frac: float) -> None:  # pragma: no cover
    log.info("[%3d%%] %s", int(frac * 100), msg)


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_chunk_rows(extracted: ExtractedPDF) -> list[dict]:
    """Passages texte + tableaux, prêts à être embeddés (sans embedding)."""
    rows: list[dict] = []
    for page, text in extracted.text_chunks:
        rows.append(
            {"kind": "text", "page": page, "content": text,
             "metadata": {"filename": extracted.filename}}
        )
    for t in extracted.tables:
        content = f"{t.caption}\n{t.markdown}" if t.caption else t.markdown
        rows.append(
            {"kind": "table", "page": t.page, "content": content,
             "metadata": {"filename": extracted.filename, "caption": t.caption}}
        )
    return rows


def ingest_pdf(path: str | Path, progress: Progress | None = None) -> dict:
    """Indexe un PDF. Ré-indexer le même fichier remplace son contenu."""
    progress = progress or _noop
    path = Path(path)
    settings.images_dir.mkdir(parents=True, exist_ok=True)

    progress("Analyse du PDF (LlamaIndex + PyMuPDF)...", 0.05)
    extracted = extract_pdf(path)
    progress(
        f"{len(extracted.text_chunks)} passages, {len(extracted.tables)} tableaux, "
        f"{len(extracted.images)} images extraits.",
        0.2,
    )

    db.init_db()
    with db.connect() as conn:
        db.replace_document(
            conn, extracted.doc_id, extracted.filename, extracted.n_pages
        )

        # --- texte + tableaux -> BGE-M3 -> chunks
        rows = build_chunk_rows(extracted)
        progress("Embeddings texte/tableaux (BGE-M3)...", 0.3)
        done = 0
        for batch in _batched(rows, 32):
            vecs = models.embed_texts([r["content"] for r in batch])
            for r, v in zip(batch, vecs):
                r["embedding"] = v
            db.insert_chunks(conn, extracted.doc_id, batch)
            done += len(batch)
            progress(f"Texte/tableaux indexés : {done}/{len(rows)}",
                     0.3 + 0.3 * done / max(len(rows), 1))

        # --- images -> description VLM + CLIP
        n_img = len(extracted.images)
        for k, img in enumerate(extracted.images, start=1):
            progress(f"Image {k}/{n_img} (page {img.page}) : description + CLIP...",
                     0.6 + 0.35 * (k - 1) / max(n_img, 1))
            description = describe_image(img.path, img.caption)
            clip_vec = models.clip_embed_images([img.path])[0]
            meta = {"filename": extracted.filename, "source": img.source,
                    "bbox": [round(x, 1) for x in img.bbox]}
            image_id = db.insert_image(
                conn, extracted.doc_id, img.page, str(img.path),
                img.caption, description, meta, clip_vec,
            )
            if description:  # la description devient interrogeable par le texte
                desc_vec = models.embed_texts([description])[0]
                db.insert_chunks(conn, extracted.doc_id, [{
                    "kind": "image_description", "page": img.page,
                    "content": description,
                    "metadata": {**meta, "image_id": image_id,
                                 "path": str(img.path)},
                    "embedding": desc_vec,
                }])

    progress("Indexation terminée.", 1.0)
    return {
        "doc_id": extracted.doc_id,
        "filename": extracted.filename,
        "pages": extracted.n_pages,
        "text_chunks": len(extracted.text_chunks),
        "tables": len(extracted.tables),
        "images": n_img,
    }
