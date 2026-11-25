# Initialisation avec uv (Windows PowerShell) : .venv + dépendances + base PGVector.
# Usage : .\init.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv introuvable : https://docs.astral.sh/uv/"; exit 1
}

Write-Host ">> uv sync (crée .venv et installe les dépendances de pyproject.toml)"
uv sync

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host ">> .env créé depuis .env.example"
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host ">> Démarrage de PostgreSQL + pgvector"
    docker compose up -d db
} else {
    Write-Host "!! Docker introuvable : lancez PostgreSQL + pgvector vous-même (voir PG_DSN dans .env)"
}

Write-Host ">> Initialisation de la base (init-db)"
$ErrorActionPreference = "Continue"
for ($i = 1; $i -le 20; $i++) {
    uv run python cli.py init-db
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "   base pas encore prête, nouvel essai ($i/20)..."
    Start-Sleep -Seconds 3
}

Write-Host @"

Terminé. Pour travailler :
  ollama pull qwen2.5:7b ; ollama pull qwen2.5vl:7b
  uv run python cli.py ingest data\samples\attention_is_all_you_need.pdf
  uv run streamlit run app.py
"@
