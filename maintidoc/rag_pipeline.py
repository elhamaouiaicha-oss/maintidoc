"""RAG « simple » (non agentique) : récupérer -> construire le contexte -> générer."""

from __future__ import annotations

from dataclasses import dataclass

import ollama

from . import retrieval
from .config import settings
from .retrieval import Hit

SYSTEM_PROMPT = (
    "Tu es MaintiDoc, assistant documentaire technique. Réponds UNIQUEMENT à "
    "partir du contexte fourni, dans la langue de la question. Cite tes sources "
    "avec leur étiquette entre crochets, par exemple [S1]. Si le contexte ne "
    "contient pas la réponse, dis-le clairement au lieu de deviner. Pour les "
    "valeurs numériques et les noms de composants, recopie-les fidèlement."
)

MAX_CHARS_PER_SOURCE = 1200


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = h.content.strip()[:MAX_CHARS_PER_SOURCE]
        blocks.append(f"[S{i}] ({h.describe()})\n{text}")
    return "\n\n".join(blocks)


@dataclass
class RagAnswer:
    text: str
    hits: list[Hit]


def answer(
    question: str,
    doc_id: str | None = None,
    history: list[dict] | None = None,
    k_text: int = 5,
    k_images: int = 2,
) -> RagAnswer:
    hits = retrieval.retrieve(question, k_text=k_text, k_images=k_images, doc_id=doc_id)
    context = format_context(hits) if hits else "(aucun passage trouvé)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-6:]:  # contexte conversationnel borné
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"}
    )
    client = ollama.Client(host=settings.ollama_host)
    resp = client.chat(
        model=settings.llm_model, messages=messages, options={"temperature": 0.1}
    )
    return RagAnswer(text=resp["message"]["content"].strip(), hits=hits)
