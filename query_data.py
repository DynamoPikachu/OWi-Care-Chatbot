import argparse
import os
import re
import time

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function


# TODO vorschläge von ChatGPT umsetzen
"""Wichtiger Praxis-Hinweis zu LM Studio + Embeddings

LM Studio hat zwar OpenAI-kompatible Endpoints inkl. Embeddings

, aber nicht jedes GGUF/Modell in LM Studio eignet sich sinnvoll als Embedding-Modell. In der Praxis ist oft stabiler:

Embeddings in Python via sentence-transformers / transformers auf der GPU berechnen (bge-m3 / e5),

Chat-Modell in LM Studio laufen lassen.

Das ist völlig normal in lokalen RAG-Setups."""

CHROMA_PATH = "chroma"

AGENT_SYSTEM_PROMPT = """
Du bist ein erfahrener Ernährungsberater, der sich auf die Ernährungstherapie von Kindern mit komplexen Grunderkrankungen und Behinderungen spezialisiert hat. Deine Fachkenntnisse beinhalten aktuelle Forschungsergebnisse zu Sondenentwöhnung und den besonderen Ernährungsbedürfnissen dieser Kinder.

Rolle: Ernährungsberater für Familien mit Kindern, die besondere Bedürfnisse haben. Du bist empathisch, informativ und äußerst geduldig. Du bietest praktische Ratschläge und individuelle Ernährungslösungen an.

Zielgruppe: Eltern von Kindern mit Behinderungen, die Unterstützung bei der Ernährungstherapie und Sondenentwöhnung benötigen.

Aufgabe: Beantworte die individuellen Frage oder gehe auf die Probleme der Eltern ein. Achte darauf, auf die besonderen Bedürfnisse des Kindes einzugehen und praktische Tipps zu geben. Gib keine Anweisungen oder Vorschläge die das Kind oder andere Beteiligte verletzen könnten. Benutze das `search_docs` tool um relevanten Kontext aus dem vector store zu finden bevor du eine Antwort gibst. Falls nötig, rufe das tool mehrmals auf.

Visualisierung bzw. Ausgabeformat: Fließtext mit hilfreichen, nachvollziehbaren Anweisungen und Empfehlungen.
Antworte wenn möglich in 2-3 Sätzen.
""".strip()


def main():
    # set parameters
    USE_LM_STUDIO = True

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", nargs="?", type=str, help="The query text.")
    parser.add_argument(
        "--input-file",
        type=str,
        help="Textdatei mit mehreren Prompts (durch Leerzeile getrennt).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Datei, in die die Ergebnisse geschrieben werden.",
    )
    args = parser.parse_args()

    if args.input_file:
        if not args.output_file:
            parser.error("--output-file ist erforderlich, wenn --input-file verwendet wird.")
        process_prompt_file(
            input_path=args.input_file,
            output_path=args.output_file,
            use_lm_studio=USE_LM_STUDIO,
        )
        return

    if not args.query_text:
        parser.error("query_text oder --input-file ist erforderlich.")

    query_rag(args.query_text, USE_LM_STUDIO)


def query_rag(query_text: str, use_lm_studio: bool = True):
    # Prepare the DB.
    embedding_platform = "lm-studio" if use_lm_studio else "ollama"
    embedding_function = get_embedding_function(embedding_platform)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    retrieved_docs = []

    @tool
    def search_docs(query: str) -> str:
        """Search the vector store for relevant context."""
        docs = db.similarity_search(query, k=5)
        retrieved_docs.extend(docs)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    if use_lm_studio:
        from langchain_openai import ChatOpenAI
        api_base = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
        model_name = os.getenv("LMSTUDIO_CHAT_MODEL", "lmstudio")
        model = ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key="lm-studio",
        )
    else:
        model = ChatOllama(model="llama3.2:3b")
    
    agent_graph = create_agent(
        model=model,
        tools=[search_docs],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    ### hier geschieht die Magie
    response = agent_graph.invoke({"messages": [HumanMessage(content=query_text)]})
    ####################################

    response_text = ""
    for message in reversed(response.get("messages", [])):
        if isinstance(message, AIMessage):
            response_text = message.content
            break
    if not response_text:
        response_text = "No response from model."

    sources = []
    seen = set()
    for doc in retrieved_docs:
        source_id = doc.metadata.get("id", None)
        if source_id and source_id not in seen:
            seen.add(source_id)
            sources.append(source_id)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text, sources


def _extract_pdf_sources(sources: list[str]) -> list[str]:
    filenames = set()
    for source in sources:
        filenames.update(re.findall(r"([^\\/]+\.pdf)", source))
    return sorted(filenames)


def _load_prompts_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()

    blocks = [block.strip() for block in re.split(r"\n\s*\n", content)]
    return [block for block in blocks if block]


def process_prompt_file(input_path: str, output_path: str, use_lm_studio: bool = True):
    prompts = _load_prompts_from_file(input_path)
    if not prompts:
        raise ValueError(f"Keine Prompts in {input_path} gefunden.")

    with open(output_path, "w", encoding="utf-8") as handle:
        for prompt in prompts:
            start_time = time.time()
            response_text, sources = query_rag(prompt, use_lm_studio)
            duration = int(time.time() - start_time)
            pdf_sources = _extract_pdf_sources(sources)

            handle.write("🧑 Du:\n")
            handle.write(prompt + "\n\n")
            handle.write("🤖 Assistant:\n")
            handle.write(response_text + "\n")
            if pdf_sources:
                handle.write(f"\n(Quellen: {', '.join(pdf_sources)})\n")
            handle.write(f"\nDauer: {duration}s\n")
            handle.write("─" * 60 + "\n\n")


if __name__ == "__main__":
    main()
