"""Accès PostgreSQL + pgvector : schéma, insertions, recherche vectorielle."""

from __future__ import annotations

from typing import Any, Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,
    n_pages    INT,
    metadata   JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Passages de texte, tableaux (Markdown) et descriptions d'images,
-- tous embeddés avec le même modèle texte (BGE-M3).
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    kind      TEXT NOT NULL CHECK (kind IN ('text', 'table', 'image_description')),
    page      INT,
    content   TEXT NOT NULL,
    metadata  JSONB DEFAULT '{}'::jsonb,
    embedding vector(__TEXT_DIM__) NOT NULL
);

-- Images : fichier PNG + embedding CLIP + description textuelle.
CREATE TABLE IF NOT EXISTS images (
    id             BIGSERIAL PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page           INT,
    path           TEXT NOT NULL,
    caption        TEXT,
    description    TEXT,
    metadata       JSONB DEFAULT '{}'::jsonb,
    clip_embedding vector(__CLIP_DIM__) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id, page);
CREATE INDEX IF NOT EXISTS chunks_emb_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS images_clip_idx
    ON images USING hnsw (clip_embedding vector_cosine_ops);
""".replace("__TEXT_DIM__", str(settings.text_embed_dim)).replace(
    "__CLIP_DIM__", str(settings.clip_dim)
)


def init_db() -> None:
    """Crée l'extension pgvector et les tables (idempotent)."""
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA)


def connect() -> psycopg.Connection:
    """Connexion avec les types pgvector enregistrés. À utiliser avec `with`."""
    conn = psycopg.connect(settings.pg_dsn, autocommit=True, row_factory=dict_row)
    register_vector(conn)
    return conn


# ----------------------------------------------------------------- écriture
def replace_document(
    conn: psycopg.Connection,
    doc_id: str,
    filename: str,
    n_pages: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """(Ré)enregistre un document ; le CASCADE efface ses anciens contenus."""
    conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
    conn.execute(
        "INSERT INTO documents (doc_id, filename, n_pages, metadata) "
        "VALUES (%s, %s, %s, %s)",
        (doc_id, filename, n_pages, Jsonb(metadata or {})),
    )


def insert_chunks(
    conn: psycopg.Connection, doc_id: str, rows: Sequence[dict[str, Any]]
) -> None:
    """rows : dicts {kind, page, content, metadata?, embedding}."""
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_id, kind, page, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (
                    doc_id,
                    r["kind"],
                    r["page"],
                    r["content"],
                    Jsonb(r.get("metadata", {})),
                    Vector(r["embedding"]),
                )
                for r in rows
            ],
        )


def insert_image(
    conn: psycopg.Connection,
    doc_id: str,
    page: int,
    path: str,
    caption: str,
    description: str,
    metadata: dict[str, Any],
    clip_vec: Sequence[float],
) -> int:
    row = conn.execute(
        "INSERT INTO images (doc_id, page, path, caption, description, metadata, "
        "clip_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (doc_id, page, path, caption, description, Jsonb(metadata), Vector(clip_vec)),
    ).fetchone()
    return int(row["id"])


# ------------------------------------------------------------------ lecture
def list_documents(conn: psycopg.Connection) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT d.doc_id, d.filename, d.n_pages, "
        "(SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id) AS n_chunks, "
        "(SELECT count(*) FROM images i WHERE i.doc_id = d.doc_id) AS n_images "
        "FROM documents d ORDER BY d.created_at DESC"
    ).fetchall()


def search_chunks(
    conn: psycopg.Connection,
    query_vec: Sequence[float],
    k: int = 5,
    doc_id: str | None = None,
    kinds: Sequence[str] = ("text", "table", "image_description"),
) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT id, doc_id, kind, page, content, metadata,
               1 - (embedding <=> %(q)s) AS score
        FROM chunks
        WHERE (%(doc)s::text IS NULL OR doc_id = %(doc)s)
          AND kind = ANY(%(kinds)s)
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": Vector(query_vec), "doc": doc_id, "kinds": list(kinds), "k": k},
    ).fetchall()


def search_images(
    conn: psycopg.Connection,
    clip_vec: Sequence[float],
    k: int = 3,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT id, doc_id, page, path, caption, description, metadata,
               1 - (clip_embedding <=> %(q)s) AS score
        FROM images
        WHERE (%(doc)s::text IS NULL OR doc_id = %(doc)s)
        ORDER BY clip_embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": Vector(clip_vec), "doc": doc_id, "k": k},
    ).fetchall()


def get_image(conn: psycopg.Connection, image_id: int) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT id, doc_id, page, path, caption, description, metadata "
        "FROM images WHERE id = %s",
        (image_id,),
    ).fetchone()


def get_page_chunks(
    conn: psycopg.Connection, doc_id: str | None, page: int
) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT id, doc_id, kind, page, content, metadata FROM chunks "
        "WHERE (%(doc)s::text IS NULL OR doc_id = %(doc)s) AND page = %(p)s "
        "ORDER BY id",
        {"doc": doc_id, "p": page},
    ).fetchall()
