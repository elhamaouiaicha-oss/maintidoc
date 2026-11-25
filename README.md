# MaintiDoc — Étape 1 : RAG multimodal agentique

Application de question-réponse sur des PDF contenant du **texte, des tableaux et des
figures**. Indexation multimodale dans **PostgreSQL + pgvector**, génération par un
**LLM open source local (Qwen via Ollama)**, orchestration **agentique LangChain**,
interface **Streamlit**.

```
PDF ──LlamaIndex──► passages ─────────┐
    ──PyMuPDF────► tableaux (MD) ─────┼─► BGE-M3 ──► chunks ─┐
    ──PyMuPDF────► images ─► VLM (description) ─► BGE-M3 ────┤  PGVector
                         └─► CLIP ───────────────► images ───┘
Streamlit ─► agent LangChain ─► outils de recherche ─► Qwen/Ollama ─► réponse + sources
```

## Correspondance avec le cahier des charges

| Exigence | Réalisation | Fichier |
|---|---|---|
| Parser le PDF avec LlamaIndex (texte, images, tableaux) | LlamaIndex (`PyMuPDFReader` + `SentenceSplitter`) pour le texte ; PyMuPDF pour tableaux et figures | `maintidoc/extraction.py` |
| Embeddings open source → PGVector | BGE-M3 (1024 d), tables `documents`, `chunks`, `images`, index HNSW | `models.py`, `db.py`, `ingestion.py` |
| CLIP + descriptions d'images indexées | CLIP `clip-ViT-B-32` ; description par VLM Qwen2.5-VL, indexée comme passage | `models.py`, `vision.py` |
| RAG avec LLM open source (Qwen/Ollama) | Récupération multimodale + fusion RRF, génération avec citations `[S#]` | `retrieval.py`, `rag_pipeline.py` |
| Interface Streamlit | Chargement, indexation, chat, sources avec images | `app.py` |
| RAG agentique LangChain | Agent à 4 outils (texte, tableaux, images, page), historique borné | `agent.py` |

## Installation

Prérequis : Python 3.11+, Docker, [Ollama](https://ollama.com).

Le projet est géré avec **uv** (`pyproject.toml`). Un script fait tout d'un coup :
`uv sync` (crée `.venv` et installe les dépendances), copie `.env`, démarre
PostgreSQL et lance `init-db`.

```bash
./init.sh            # Linux / macOS  (chmod +x init.sh si besoin)
.\init.ps1           # Windows PowerShell
```

Ou à la main :

```bash
uv sync                                   # .venv + dépendances
docker compose up -d                      # PostgreSQL + pgvector
ollama pull qwen2.5:7b                    # génération + appel d'outils
ollama pull qwen2.5vl:7b                  # description des images
cp .env.example .env                      # optionnel : modèles / DSN

uv run python cli.py init-db
uv run python cli.py ingest data/samples/attention_is_all_you_need.pdf
uv run python cli.py ask "Combien de têtes d'attention utilise le Transformer ?"
uv run streamlit run app.py
```

Dans un projet `uv init` existant, copiez simplement le dossier `maintidoc/`, `app.py`,
`cli.py` et ce `pyproject.toml` (ou `uv add` les dépendances listées dedans).

Au premier lancement, `sentence-transformers` télécharge BGE-M3 (~2,3 Go) et CLIP.
Sans VLM, l'indexation continue : les images restent cherchables par CLIP et par leur
légende (« Figure N ») détectée dans le PDF.

## Tests

```bash
uv run pytest -q    # extraction du PDF d'exemple, conversion de tableaux, fusion RRF
```

Les tests ne nécessitent ni PostgreSQL, ni Ollama, ni GPU.

## Organisation

```
maintidoc/
  config.py      paramètres (.env)
  extraction.py  PDF -> texte, tableaux, images (sans dépendance modèle/DB)
  vision.py      description d'image par VLM (Ollama)
  models.py      BGE-M3 et CLIP
  db.py          schéma PGVector, insertions, recherches
  ingestion.py   pipeline d'indexation
  retrieval.py   recherche texte/tableaux/images + fusion RRF
  rag_pipeline.py  RAG simple
  agent.py       RAG agentique LangChain
app.py           interface Streamlit
cli.py           init-db / ingest / ask / docs
NOTE_TECHNIQUE.md
```

## Préparer les étapes suivantes

- `metadata JSONB` sur `documents` et `chunks` : prêt pour versions et droits d'accès
  (Étape 1 de la feuille de route).
- `db.py` concentre tout l'accès à la base : migrer vers Qdrant se limite à ce module.
- `agent.py` : les outils sont des fonctions isolées ; l'Étape 3 (LangGraph, MCP)
  ajoutera `get_equipment_history` / `create_ticket` au même titre.
