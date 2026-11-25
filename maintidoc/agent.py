"""RAG agentique avec LangChain.

L'agent (Qwen via Ollama, appel d'outils) décide lui-même :
- quel outil de recherche utiliser (texte, tableaux, images, page entière),
- combien de recherches enchaîner, avec quelles reformulations,
- quand il a assez de contexte pour répondre.

Chaque outil enregistre ses résultats dans un `SourceBook` qui attribue des
étiquettes stables [S1], [S2]... : l'agent les cite, l'interface les affiche.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from . import retrieval
from .config import settings
from .retrieval import Hit

AGENT_PROMPT = """Tu es MaintiDoc, assistant documentaire technique pour la maintenance.

Méthode :
1. Analyse la question, puis choisis le ou les outils adaptés :
   - rechercher_texte : explications, définitions, procédures, résultats ;
   - rechercher_tableaux : valeurs chiffrées, comparatifs, paramètres ;
   - rechercher_images : schémas, figures, photographies, architectures ;
   - lire_page : quand tu connais déjà la page à consulter.
2. Si les résultats sont insuffisants, reformule la requête et recherche à nouveau
   (plusieurs appels sont permis, 4 maximum).
3. Réponds UNIQUEMENT à partir des résultats d'outils, dans la langue de la
   question. Cite chaque information avec son étiquette, par exemple [S2].
4. Si l'information est absente des documents, dis-le franchement. N'invente rien,
   ne donne jamais de valeur chiffrée qui ne figure pas dans les sources.
5. Pour une figure, appuie-toi sur sa description et sa légende ; mentionne la page.
6. Tu DOIS appeler au moins un outil de recherche avant de répondre, même si tu
   crois connaître la réponse. Ne cite jamais une étiquette [S#] qu'un outil ne
   t'a pas renvoyée.
7. Pour un tableau, recopie TOUTES les lignes pertinentes pour la question (par
   exemple le modèle de base ET le modèle big) avec leurs valeurs exactes.
8. Ne commente que ce que les sources affirment explicitement ; n'ajoute aucune
   explication tirée de tes connaissances générales."""

MAX_CHARS = 1500


class SourceBook:
    """Collecte les résultats et leur attribue des étiquettes S1, S2, ..."""

    def __init__(self) -> None:
        self._labels: dict = {}
        self.hits: list[Hit] = []

    def label(self, hit: Hit) -> str:
        if hit.key not in self._labels:
            self._labels[hit.key] = f"S{len(self._labels) + 1}"
            self.hits.append(hit)
        return self._labels[hit.key]

    def format(self, hits: list[Hit]) -> str:
        if not hits:
            return "Aucun résultat pertinent."
        out = []
        for h in hits:
            body = h.content.strip()[:MAX_CHARS]
            out.append(f"[{self.label(h)}] ({h.describe()})\n{body}")
        return "\n\n".join(out)


def make_tools(book: SourceBook, doc_id: str | None) -> list:
    @tool
    def rechercher_texte(requete: str) -> str:
        """Recherche sémantique dans les passages de TEXTE du document
        (explications, définitions, méthodes, conclusions)."""
        return book.format(
            retrieval.search_passages(requete, k=5, doc_id=doc_id, kinds=("text",))
        )

    @tool
    def rechercher_tableaux(requete: str) -> str:
        """Recherche dans les TABLEAUX du document (valeurs chiffrées, scores,
        hyperparamètres, comparaisons)."""
        return book.format(
            retrieval.search_passages(requete, k=3, doc_id=doc_id, kinds=("table",))
        )

    @tool
    def rechercher_images(requete: str) -> str:
        """Recherche des IMAGES (schémas, figures, photos) à partir d'une
        description textuelle ; renvoie légende et description de chaque figure."""
        return book.format(retrieval.search_images(requete, k=3, doc_id=doc_id))

    @tool
    def lire_page(page: int) -> str:
        """Retourne tout le contenu indexé d'une page précise du document."""
        return book.format(retrieval.get_page(int(page), doc_id=doc_id))

    return [rechercher_texte, rechercher_tableaux, rechercher_images, lire_page]


def _create_agent(model, tools):
    try:  # LangChain >= 1.0
        from langchain.agents import create_agent

        return create_agent(model, tools, system_prompt=AGENT_PROMPT)
    except ImportError:  # anciennes versions : agent ReAct de LangGraph
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model, tools, prompt=AGENT_PROMPT)


@dataclass
class AgentResult:
    text: str
    hits: list[Hit]
    steps: list[str] = field(default_factory=list)  # appels d'outils, pour l'UI


def run_agent(
    question: str,
    doc_id: str | None = None,
    history: list[dict] | None = None,
) -> AgentResult:
    book = SourceBook()
    model = ChatOllama(
        model=settings.llm_model, base_url=settings.ollama_host, temperature=0.1
    )
    agent = _create_agent(model, make_tools(book, doc_id))

    messages = [(m["role"], m["content"]) for m in (history or [])[-6:]]
    messages.append(("user", question))
    result = agent.invoke(
        {"messages": messages}, config={"recursion_limit": 14}
    )

    steps = []
    for m in result["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            steps.append(f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")

    if not book.hits:
        # Garde-fou : un petit modèle local oublie parfois d'appeler les outils et
        # répond de mémoire avec de fausses citations. On force alors une
        # récupération classique pour que la réponse repose sur le document.
        from .rag_pipeline import answer as simple_answer

        forced = simple_answer(question, doc_id=doc_id, history=history)
        steps.append("aucun outil appelé par l'agent -> recherche forcée")
        return AgentResult(text=forced.text, hits=forced.hits, steps=steps)

    final = result["messages"][-1].content
    if isinstance(final, list):  # contenu par blocs
        final = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in final)
    return AgentResult(text=final.strip(), hits=book.hits, steps=steps)