#!/usr/bin/env bash
# Initialisation avec uv : .venv + dépendances + base PGVector.
# Usage : ./init.sh
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || { echo "uv introuvable : https://docs.astral.sh/uv/"; exit 1; }

echo ">> uv sync (crée .venv et installe les dépendances de pyproject.toml)"
uv sync

[ -f .env ] || { cp .env.example .env; echo ">> .env créé depuis .env.example"; }

if command -v docker >/dev/null 2>&1; then
  echo ">> Démarrage de PostgreSQL + pgvector"
  docker compose up -d db
else
  echo "!! Docker introuvable : lancez PostgreSQL + pgvector vous-même (voir PG_DSN dans .env)"
fi

echo ">> Initialisation de la base (init-db)"
for i in $(seq 1 20); do
  if uv run python cli.py init-db; then break; fi
  echo "   base pas encore prête, nouvel essai ($i/20)..."; sleep 3
done

cat <<'EOF'

Terminé. Pour travailler :
  ollama pull qwen2.5:7b && ollama pull qwen2.5vl:7b
  uv run python cli.py ingest data/samples/attention_is_all_you_need.pdf
  uv run streamlit run app.py
EOF
