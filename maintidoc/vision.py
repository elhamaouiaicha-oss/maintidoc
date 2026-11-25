"""Description textuelle des images par un VLM open source servi par Ollama."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "Tu analyses une figure extraite d'un document technique (schéma, courbe, "
    "photographie, diagramme). Décris-la en 3 à 6 phrases : type de figure, "
    "éléments principaux, texte lisible (étiquettes, légendes, valeurs) et "
    "relations ou flux entre les éléments. Reste factuel : n'invente rien qui "
    "ne soit pas visible."
)


def describe_image(path: str | Path, caption: str = "") -> str:
    """Retourne une description indexable : légende du PDF + description du VLM.

    Si le VLM est indisponible (Ollama éteint, modèle non téléchargé), on
    retombe sur la seule légende : l'image reste cherchable via CLIP.
    """
    vlm_text = ""
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        prompt = DESCRIBE_PROMPT
        if caption:
            prompt += f"\nLégende présente dans le document : {caption}"
        resp = client.chat(
            model=settings.vlm_model,
            messages=[{"role": "user", "content": prompt, "images": [str(path)]}],
            options={"temperature": 0.1},
        )
        vlm_text = resp["message"]["content"].strip()
    except Exception as exc:
        log.warning("Description VLM indisponible pour %s : %s", path, exc)

    parts = [p for p in (caption, vlm_text) if p]
    return "\n".join(parts)
