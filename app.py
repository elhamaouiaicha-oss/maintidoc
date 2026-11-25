"""Interface Streamlit de MaintiDoc : chargement du PDF, questions, réponses sourcées.

Lancer :  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from maintidoc import db
from maintidoc.config import settings

st.set_page_config(page_title="MaintiDoc", page_icon="🛠️", layout="wide")


@st.cache_resource
def _init() -> bool:
    db.init_db()
    return True


def _documents() -> list[dict]:
    with db.connect() as conn:
        return db.list_documents(conn)


def render_sources(hits: list[dict]) -> None:
    """Affiche les sources citées : texte, tableaux et images avec leur page."""
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})", expanded=False):
        for i, h in enumerate(hits, start=1):
            label = {"text": "Texte", "table": "Tableau", "image": "Image"}[h["kind"]]
            st.markdown(f"**[S{i}]** {h['filename']} — page {h['page']} — {label}")
            if h["kind"] == "image" and h.get("image_path") and Path(h["image_path"]).exists():
                st.image(h["image_path"], width=420)
            if h["kind"] == "table":
                st.code(h["content"], language=None)
            else:
                st.caption(h["content"][:700])


def main() -> None:
    try:
        _init()
    except Exception as exc:
        st.error(
            "Base PostgreSQL inaccessible. Lance `docker compose up -d` puis "
            f"recharge la page.\n\nDétail : {exc}"
        )
        st.stop()

    st.title("MaintiDoc — RAG multimodal agentique")
    st.caption("Texte, tableaux et figures de vos PDF — LLM local (Ollama), index PGVector.")

    # ---------------------------------------------------------------- barre latérale
    with st.sidebar:
        st.header("1. Document")
        uploaded = st.file_uploader("Charger un PDF", type=["pdf"])
        if uploaded is not None and st.button("Indexer ce PDF", type="primary"):
            from maintidoc.ingestion import ingest_pdf

            settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            target = settings.uploads_dir / uploaded.name
            target.write_bytes(uploaded.getbuffer())
            bar = st.progress(0.0)
            status = st.empty()

            def progress(msg: str, frac: float) -> None:
                bar.progress(min(frac, 1.0))
                status.caption(msg)

            try:
                stats = ingest_pdf(target, progress)
                st.success(
                    f"{stats['text_chunks']} passages, {stats['tables']} tableaux, "
                    f"{stats['images']} images indexés."
                )
                st.session_state["doc_id"] = stats["doc_id"]
            except Exception as exc:
                st.error(f"Échec de l'indexation : {exc}")

        docs = _documents()
        options = {"Tous les documents": None}
        options.update(
            {f"{d['filename']} ({d['n_pages']} p., {d['n_images']} img.)": d["doc_id"] for d in docs}
        )
        labels = list(options)
        default = 0
        if st.session_state.get("doc_id"):
            for i, lab in enumerate(labels):
                if options[lab] == st.session_state["doc_id"]:
                    default = i
        choice = st.selectbox("Document interrogé", labels, index=default)
        doc_id = options[choice]

        st.header("2. Mode")
        mode = st.radio(
            "Pipeline",
            ["Agent LangChain", "RAG simple"],
            help="L'agent choisit ses outils de recherche et peut enchaîner plusieurs "
            "requêtes ; le RAG simple fait une seule récupération.",
        )
        st.caption(f"LLM : `{settings.llm_model}` · VLM : `{settings.vlm_model}`")
        if st.button("Effacer la conversation"):
            st.session_state["messages"] = []

    # ------------------------------------------------------------------ conversation
    st.session_state.setdefault("messages", [])
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("steps"):
                st.caption("Outils appelés : " + " → ".join(m["steps"]))
            render_sources(m.get("sources", []))

    question = st.chat_input("Posez votre question sur le document…")
    if not question:
        return
    if not docs:
        st.warning("Indexez d'abord un PDF dans la barre latérale.")
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération en cours…"):
            try:
                if mode == "Agent LangChain":
                    from maintidoc.agent import run_agent

                    res = run_agent(question, doc_id=doc_id, history=history)
                    steps = res.steps
                else:
                    from maintidoc.rag_pipeline import answer

                    res = answer(question, doc_id=doc_id, history=history)
                    steps = []
                text, hits = res.text, [h.to_dict() for h in res.hits]
            except Exception as exc:
                text, hits, steps = f"Erreur : {exc}\n\nOllama est-il lancé et les modèles téléchargés ?", [], []
        st.markdown(text)
        if steps:
            st.caption("Outils appelés : " + " → ".join(steps))
        render_sources(hits)

    st.session_state["messages"].append(
        {"role": "assistant", "content": text, "sources": hits, "steps": steps}
    )


main()
