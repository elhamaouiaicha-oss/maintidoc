# Note technique — MaintiDoc, Étape 1 : RAG multimodal agentique

## 1. Objectif et périmètre

Répondre à des questions sur un PDF mêlant texte, tableaux et figures, avec des
réponses fondées sur des passages retrouvés et citées par page. Document de test :
« Attention Is All You Need » (11 pages, 3 tableaux, 3 figures détectées).
Hors périmètre de cette étape : historique d'interventions, tickets, MCP, droits
d'accès, bots (Étapes 3 et 4 de la feuille de route).

## 2. Modèles et outils

| Rôle | Choix | Justification |
|---|---|---|
| Parsing texte | LlamaIndex `PyMuPDFReader` + `SentenceSplitter` | Exigé par le cahier des charges ; texte par page, pages conservées en métadonnées |
| Tableaux, figures | PyMuPDF (`find_tables`, `cluster_drawings`) | LlamaIndex seul n'extrait pas figures/tableaux hors service cloud (LlamaParse) |
| Embeddings texte | `BAAI/bge-m3` (1024 d) | Open source, multilingue FR/EN : question en français sur un document anglais |
| Embeddings image | CLIP `clip-ViT-B-32` (512 d) | Espace commun texte ↔ image |
| Description d'images | `qwen2.5vl:7b` via Ollama | VLM open source ; la description est indexée comme un passage |
| Base vectorielle | PostgreSQL 16 + pgvector, index HNSW cosinus | Exigé ; filtrage SQL par document et par type de contenu |
| Génération | `qwen2.5:7b` via Ollama | Open source, appel d'outils supporté |
| Orchestration | LangChain (`create_agent`), `langchain-ollama` | Exigé ; agent à 4 outils |
| Interface | Streamlit | Exigé |

## 3. Architecture de l'index

- `chunks` : passages texte (`text`), tableaux (`table`) et descriptions d'images
  (`image_description`), tous embeddés par BGE-M3, avec page et métadonnées JSONB.
- `images` : fichier PNG recadré, légende détectée, description, embedding CLIP.

Une figure est donc retrouvable de deux façons : par CLIP (requête → image) et par sa
description (requête → texte). `retrieval.search_images` fusionne les deux classements
par Reciprocal Rank Fusion, car les scores de deux modèles différents ne sont pas
comparables.

## 4. Orchestration agentique

L'agent dispose de `rechercher_texte`, `rechercher_tableaux`, `rechercher_images` et
`lire_page`. Il choisit les outils, peut reformuler et enchaîner les recherches
(limite de récursion fixée), puis répond en citant `[S#]`. Gestion du contexte : les 6
derniers messages sont transmis ; chaque résultat reçoit une étiquette stable
(`SourceBook`) que l'interface relie aux passages, tableaux et images affichés. Le mode
« RAG simple » (une récupération, une génération) reste disponible en comparaison ou en
repli si le modèle local appelle mal les outils.

## 5. Vérifications effectuées

- Extraction testée sur le PDF d'exemple (`pytest`, 4 tests) : 3 tableaux avec légendes
  (Tables 1 à 3) et nombres décimaux intacts, 3 figures (Figure 1 et les deux schémas de
  la Figure 2) rendues en PNG, texte présent sur les 11 pages.
- Modules compilés ; import de l'agent vérifié avec LangChain 1.4 (`create_agent`).
- **Non exécuté de bout en bout** : l'environnement de rédaction ne disposait ni
  d'Ollama, ni de PostgreSQL, ni des poids de BGE-M3/CLIP. L'indexation complète, la
  recherche vectorielle SQL, la génération et l'interface Streamlit n'ont donc pas été
  lancées ; la première exécution chez vous peut révéler des ajustements (versions de
  bibliothèques, noms de modèles Ollama).

## 6. Exemples de questions / réponses

Les réponses attendues sont tirées du papier lui-même (section indiquée). La colonne
« observé » est à remplir après vos essais : je n'ai pas pu exécuter le LLM.

| # | Question | Réponse attendue (source) | Outil attendu | Observé |
|---|---|---|---|---|
| 1 | Combien de têtes d'attention et quelles dimensions ? | h = 8, d_k = d_v = 64, d_model = 512 (§3.2.2) | texte | à compléter |
| 2 | Quels scores BLEU obtient le Transformer (big) ? | 28,4 en EN→DE et 41,0 en EN→FR (Table 2) | tableaux | à compléter |
| 3 | Que représente la Figure 2 (gauche) ? | Attention « scaled dot-product » : MatMul → Scale → Mask (optionnel) → SoftMax → MatMul sur Q, K, V (p. 4) | images | à compléter |
| 4 | Pourquoi diviser les produits scalaires par √d_k ? | Pour éviter que le softmax tombe dans des zones à gradients très faibles quand d_k est grand (§3.2.1) | texte | à compléter |
| 5 | Quel est l'effet d'une seule tête d'attention ? | BLEU 24,9 contre 25,8 pour le modèle de base, soit 0,9 de moins (Table 3 ligne A, §6.2) | tableaux + texte | à compléter |
| 6 | Quel est le prix d'un GPU P100 ? | Information absente : le système doit le dire sans inventer | tous | à compléter |

La question 6 est un test de non-hallucination : une bonne réponse reconnaît l'absence
d'information dans le document.

## 7. Limites connues

- **Tableaux** : les exposants sont aplatis par le PDF (« 10^20 » devient « 1020 » dans
  la Table 2) ; les tableaux sans filets sont reconstruits ligne par ligne, pas cellule
  par cellule. Les tableaux scannés (images) ne sont pas lus : pas d'OCR à cette étape.
- **Figures** : les schémas vectoriels sont repérés par une heuristique de regroupement
  de tracés (taille minimale, exclusion des filets de tableau) ; sur d'autres documents,
  des figures peuvent être scindées, fusionnées ou manquées.
- **CLIP ViT-B-32** est surtout entraîné sur l'anglais et tronque les requêtes à 77
  tokens ; pour le français, les descriptions indexées avec BGE-M3 compensent en partie.
- **VLM et LLM de 7 milliards de paramètres** : ils peuvent mal lire un petit texte dans
  une figure ou halluciner ; la légende du PDF est ajoutée à la description pour limiter
  ce risque, et la consigne impose de citer les sources.
- **Agent local** : un modèle 7B peut choisir un outil inadapté ou ne pas en appeler ; le
  mode RAG simple sert alors de repli.
- **Découpage en mots** (et non en tokens) : approximation volontaire, qui évite de
  télécharger un tokenizer ; les passages restent bien sous la limite de 512 tokens de
  BGE-M3 avec les valeurs par défaut.
- **Aucune évaluation chiffrée** de la qualité de récupération ou de réponse n'a été
  réalisée ; elle relève de l'Étape 2 (nDCG@5, MRR@10) de la feuille de route.
- Pas de gestion des droits d'accès ni de versions de documents (prévus ensuite).


| 1 | Combien de têtes d'attention et quelles dimensions ? | h = 8, d_k = d_v = 64, d_model = 512 (§3.2.2) | texte | ⚠️ 8 têtes et d_model = 512 corrects, ajoute le modèle big (16 têtes, d_model = 1024) ; d_k = d_v = 64 non mentionnés ([S2], p.X, outil Y) |

| 2 | Quels scores BLEU obtient le Transformer (big) ? | 28,4 en EN→DE et 41,0 en EN→FR (Table 2) | tableaux | ✅ 28.4 et 41.0 corrects, cités [S1] (Table 2, p.8) ; limite visible : exposants aplatis (« 1020 » pour 10^20) |

| 3 | Que représente la Figure 2 (gauche) ? | Attention « scaled dot-product » : MatMul → Scale → Mask (optionnel) → SoftMax → MatMul sur Q, K, V (p. 4) | images | ⚠️ figure correcte retrouvée ([S7], p.4) et nommée « Scaled Dot-Product Attention », mais étapes MatMul/Scale/Mask/SoftMax absentes : la description VLM a surtout décrit la partie droite (Multi-Head) ; une phrase générale non sourcée ajoutée par le modèle |

| 4 | Pourquoi diviser les produits scalaires par √d_k ? | Pour éviter que le softmax tombe dans des zones à gradients très faibles quand d_k est grand (§3.2.1) | texte | ⚠️ formule et idée générale (stabilité du softmax quand d_k grandit) correctes, mais « gradients très faibles » absent, formulation vague ; citation [S1] qui ne contient pas l'explication (p.4) : passage clé non retrouvé |

| 5 | Quel est l'effet d'une seule tête d'attention ? | BLEU 24,9 contre 25,8 pour le modèle de base, soit 0,9 de moins (Table 3 ligne A, §6.2) | tableaux + texte | ⚠️ explication qualitative correcte (§3.2.2), sans valeur BLEU ni citation [S#] ; l'agent n'a appelé que rechercher_texte (jamais rechercher_tableaux) ; avec l'historique non effacé, la réponse recopiait la précédente |

| 6 | Quel est le prix d'un GPU P100 ? | Information absente : le système doit le dire sans inventer | tous | ✅ reconnaît que le prix n'est pas dans le document, aucune valeur inventée (formulation limitée aux « tableaux » alors que le texte aurait aussi pu être cité) |