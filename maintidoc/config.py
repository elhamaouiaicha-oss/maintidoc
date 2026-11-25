"""Configuration centrale (variables d'environnement / fichier .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv est optionnel pour les tests purs
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # --- PostgreSQL + pgvector
    pg_dsn: str = os.getenv(
        "PG_DSN", "postgresql://maintidoc:maintidoc@localhost:5432/maintidoc"
    )

    # --- Ollama (LLM de génération + VLM pour décrire les images)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    vlm_model: str = os.getenv("VLM_MODEL", "qwen2.5vl:7b")

    # --- Embeddings texte (open source, multilingue) : BGE-M3 -> 1024 dims
    text_embed_model: str = os.getenv("TEXT_EMBED_MODEL", "BAAI/bge-m3")
    text_embed_dim: int = int(os.getenv("TEXT_EMBED_DIM", "1024"))

    # --- CLIP (texte <-> image) : clip-ViT-B-32 -> 512 dims
    clip_model: str = os.getenv("CLIP_MODEL", "clip-ViT-B-32")
    clip_dim: int = int(os.getenv("CLIP_DIM", "512"))

    # --- Découpage en passages (unité : mots, voir ingestion.parse_text)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))

    # --- Stockage local des fichiers
    data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT / "data")))

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
