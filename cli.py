"""Ligne de commande : init de la base, indexation, questions.

    python cli.py init-db
    python cli.py ingest "attention_is_all_you_need.pdf"
    python cli.py ask "Combien de têtes d'attention utilise le Transformer ?"
    python cli.py ask --simple "Que montre la Figure 2 ?"
    python cli.py docs
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="maintidoc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="crée l'extension pgvector et les tables")
    sub.add_parser("docs", help="liste les documents indexés")

    p_ing = sub.add_parser("ingest", help="indexe un PDF")
    p_ing.add_argument("pdf")

    p_ask = sub.add_parser("ask", help="pose une question")
    p_ask.add_argument("question")
    p_ask.add_argument("--doc", default=None, help="doc_id à interroger")
    p_ask.add_argument("--simple", action="store_true", help="RAG simple (sans agent)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "init-db":
        from maintidoc import db

        db.init_db()
        print("Base initialisée.")
    elif args.cmd == "docs":
        from maintidoc import db

        with db.connect() as conn:
            for d in db.list_documents(conn):
                print(f"{d['doc_id']}  {d['filename']}  {d['n_pages']} p.  "
                      f"{d['n_chunks']} chunks  {d['n_images']} images")
    elif args.cmd == "ingest":
        from maintidoc.ingestion import ingest_pdf

        print(ingest_pdf(args.pdf))
    elif args.cmd == "ask":
        if args.simple:
            from maintidoc.rag_pipeline import answer

            res = answer(args.question, doc_id=args.doc)
            steps = []
        else:
            from maintidoc.agent import run_agent

            res = run_agent(args.question, doc_id=args.doc)
            steps = res.steps
        if steps:
            print("Outils :", " -> ".join(steps), "\n")
        print(res.text, "\n")
        for i, h in enumerate(res.hits, start=1):
            print(f"[S{i}] {h.describe()}  (score {h.score:.3f})")


if __name__ == "__main__":
    main()
