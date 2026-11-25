"""Extraction du contenu d'un PDF : texte (LlamaIndex), tableaux et images (PyMuPDF).

Ce module ne dépend ni de la base de données ni des modèles d'embedding :
il peut être testé seul.

- Texte    : LlamaIndex (PyMuPDFReader + SentenceSplitter) -> passages par page.
- Tableaux : PyMuPDF `find_tables()` -> Markdown (+ légende "Table N" détectée).
- Images   : figures raster intégrées ET figures vectorielles (schémas dessinés
             en traits, très fréquents dans les PDF scientifiques/techniques),
             rendues en PNG recadrés (+ légende "Figure N" détectée).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (l'alias "fitz" est déprécié)

from .config import settings

log = logging.getLogger(__name__)

MIN_FIG_W, MIN_FIG_H = 45.0, 45.0  # taille minimale (points PDF) d'une figure
MIN_FIG_AREA = 4000.0  # filtre les petits ornements (puces, filets, logos)
RENDER_DPI = 150


@dataclass
class ExtractedTable:
    page: int
    index: int
    markdown: str
    caption: str
    bbox: tuple[float, float, float, float]


@dataclass
class ExtractedImage:
    page: int
    index: int
    path: Path
    caption: str
    source: str  # "embedded" | "vector"
    bbox: tuple[float, float, float, float]


@dataclass
class ExtractedPDF:
    doc_id: str
    filename: str
    n_pages: int
    text_chunks: list[tuple[int, str]] = field(default_factory=list)  # (page, texte)
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


# --------------------------------------------------------------- utilitaires
def compute_doc_id(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Convertit une grille de cellules en tableau Markdown ('' si inexploitable)."""
    cleaned = [[_norm(c or "") for c in row] for row in rows]
    cleaned = [r for r in cleaned if any(r)]
    if len(cleaned) < 2:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    # supprime les colonnes entièrement vides
    keep = [j for j in range(width) if any(r[j] for r in cleaned)]
    if len(keep) < 2:
        return ""
    cleaned = [[r[j] for j in keep] for r in cleaned]
    header, body = cleaned[0], cleaned[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def table_text_by_lines(
    page: fitz.Page, bbox: fitz.Rect, y_tol: float = 3.0, col_gap: float = 10.0
) -> str:
    """Texte d'un tableau ligne par ligne ; colonnes séparées par ' | '.

    Plus fiable que les cellules détectées quand le tableau n'a pas de filets
    complets : chaque ligne visuelle reste une ligne, les nombres restent entiers.
    """
    words = [w for w in page.get_text("words") if bbox.intersects(fitz.Rect(w[:4]))]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list[tuple]] = []
    last_y = None
    for w in words:
        yc = (w[1] + w[3]) / 2
        if last_y is None or abs(yc - last_y) > y_tol:
            lines.append([])
        lines[-1].append(w)
        last_y = yc
    rendered = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        parts, prev_x1 = [], None
        for w in line:
            if prev_x1 is not None and w[0] - prev_x1 > col_gap:
                parts.append(" | ")
            elif prev_x1 is not None:
                parts.append(" ")
            parts.append(w[4])
            prev_x1 = w[2]
        text = "".join(parts).strip()
        if text:
            rendered.append(text)
    return "\n".join(rendered) if len(rendered) >= 2 else ""


def _nearest_caption(
    page: fitz.Page, bbox: fitz.Rect, prefixes: tuple[str, ...], prefer: str
) -> str:
    """Bloc de texte commençant par 'Figure'/'Table' le plus proche de bbox."""
    best, best_d = "", 1e9
    for b in page.get_text("blocks"):
        if b[6] != 0:  # 0 = bloc de texte
            continue
        text = _norm(b[4])
        if not text.lower().startswith(prefixes):
            continue
        y0, y1 = b[1], b[3]
        if y0 >= bbox.y1:  # sous la zone
            d = y0 - bbox.y1 + (0 if prefer == "below" else 40)
        elif y1 <= bbox.y0:  # au-dessus de la zone
            d = bbox.y0 - y1 + (0 if prefer == "above" else 40)
        else:
            d = 0
        if d < best_d:
            best, best_d = text, d
    return best[:400] if best_d < 140 else ""


def _overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return inter.get_area() / max(min(a.get_area(), b.get_area()), 1e-6)


# -------------------------------------------------------------------- texte
def parse_text(path: str | Path) -> list[tuple[int, str]]:
    """LlamaIndex : lecture page par page puis découpage en passages.

    Le tokenizer du SentenceSplitter est un simple découpage par mots : cela
    évite de télécharger un tokenizer (tiktoken) et fonctionne hors ligne.
    """
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.file import PyMuPDFReader

    docs = PyMuPDFReader().load_data(Path(path))  # un Document par page
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        tokenizer=lambda t: t.split(),
    )
    out: list[tuple[int, str]] = []
    for node in splitter.get_nodes_from_documents(docs):
        text = node.get_content().strip()
        if len(text) < 30:
            continue
        page = int(node.metadata.get("source", 0) or 0)
        out.append((page, text))
    return out


# ----------------------------------------------------------------- tableaux
def extract_tables(page: fitz.Page, pno: int) -> list[ExtractedTable]:
    out: list[ExtractedTable] = []
    try:
        finder = page.find_tables()
    except Exception as exc:  # find_tables peut échouer sur des pages atypiques
        log.warning("find_tables a échoué page %d : %s", pno, exc)
        return out
    for i, table in enumerate(finder.tables):
        raw_rows = table.extract()
        bbox = fitz.Rect(table.bbox)
        merged = any("\n" in (c or "") for row in raw_rows for c in row)
        if merged:
            # Grille mal détectée (tableaux sans filets verticaux, fréquents dans
            # les articles) : on reconstruit les lignes à partir des mots.
            md = table_text_by_lines(page, bbox)
        else:
            md = rows_to_markdown(raw_rows)
        if not md:
            continue
        caption = _nearest_caption(page, bbox, ("table",), prefer="above")
        out.append(ExtractedTable(pno, i, md, caption, tuple(bbox)))
    return out


# ------------------------------------------------------------------- images
def extract_images(
    page: fitz.Page,
    pno: int,
    out_dir: Path,
    table_boxes: list[fitz.Rect],
) -> list[ExtractedImage]:
    page_rect = page.rect
    candidates: list[tuple[str, fitz.Rect]] = []

    # 1) images raster intégrées
    embedded: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            if r.width >= MIN_FIG_W and r.height >= MIN_FIG_H:
                if r.get_area() < 0.9 * page_rect.get_area():  # ignore les fonds
                    embedded.append(r)
                    candidates.append(("embedded", r))

    # 2) figures vectorielles : regroupement des tracés proches
    try:
        clusters = page.cluster_drawings()
    except Exception:  # anciennes versions de PyMuPDF
        clusters = []
    for r in clusters:
        if r.width < MIN_FIG_W or r.height < MIN_FIG_H or r.get_area() < MIN_FIG_AREA:
            continue
        if r.get_area() > 0.9 * page_rect.get_area():
            continue
        if any(_overlap_ratio(r, t) > 0.5 for t in table_boxes):
            continue  # filets de tableau, pas une figure
        if any(_overlap_ratio(r, e) > 0.5 for e in embedded):
            continue
        candidates.append(("vector", r))

    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []
    for idx, (source, rect) in enumerate(candidates):
        clip = fitz.Rect(rect.x0 - 4, rect.y0 - 4, rect.x1 + 4, rect.y1 + 4) & page_rect
        pix = page.get_pixmap(clip=clip, dpi=RENDER_DPI)
        path = out_dir / f"p{pno:03d}_{idx}.png"
        pix.save(str(path))
        caption = _nearest_caption(page, rect, ("figure", "fig."), prefer="below")
        images.append(ExtractedImage(pno, idx, path, caption, source, tuple(rect)))
    return images


# ------------------------------------------------------------------ pipeline
def extract_pdf(path: str | Path, images_root: Path | None = None) -> ExtractedPDF:
    path = Path(path)
    doc_id = compute_doc_id(path)
    images_root = images_root or settings.images_dir
    result = ExtractedPDF(doc_id=doc_id, filename=path.name, n_pages=0)

    result.text_chunks = parse_text(path)

    with fitz.open(path) as pdf:
        result.n_pages = pdf.page_count
        for page in pdf:
            pno = page.number + 1
            tables = extract_tables(page, pno)
            result.tables.extend(tables)
            result.images.extend(
                extract_images(
                    page,
                    pno,
                    images_root / doc_id,
                    [fitz.Rect(t.bbox) for t in tables],
                )
            )
    return result
