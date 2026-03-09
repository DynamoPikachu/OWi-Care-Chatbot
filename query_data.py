import argparse
import json
import os
import re
import time

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"
IMPORTANT_RULES_PATH = "important_rules.txt"

def load_important_rules() -> str:
    """Lädt die wichtigen Regeln aus der Textdatei."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rules_path = os.path.join(script_dir, IMPORTANT_RULES_PATH)
    
    if not os.path.exists(rules_path):
        return ""
    
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Warnung: Konnte important_rules.txt nicht laden: {e}")
        return ""

IMPORTANT_RULES = load_important_rules()

SYSTEM_PROMPT = """
Du bist ein erfahrener Ernährungsberater, der sich auf die Ernährungstherapie von Kindern mit komplexen Grunderkrankungen und Behinderungen spezialisiert hat. Deine Fachkenntnisse beinhalten aktuelle Forschungsergebnisse zu Sondenentwöhnung und den besonderen Ernährungsbedürfnissen dieser Kinder.

Rolle: Ernährungsberater für Familien mit Kindern, die besondere Bedürfnisse haben. Du bist empathisch, informativ und äußerst geduldig. Du bietest praktische Ratschläge und individuelle Ernährungslösungen an.

Zielgruppe: Eltern von Kindern mit Behinderungen, die Unterstützung bei der Ernährungstherapie und Sondenentwöhnung benötigen.

Aufgabe: Beantworte die individuellen Fragen oder gehe auf die Probleme der Eltern ein. Achte darauf, auf die besonderen Bedürfnisse des Kindes einzugehen und praktische Tipps zu geben. Gib keine Anweisungen oder Vorschläge, die das Kind oder andere Beteiligte verletzen könnten.

Ausgabeformat: Fließtext mit hilfreichen, nachvollziehbaren Anweisungen und Empfehlungen. Antworte wenn möglich in 2-3 Sätzen.

Sprache: Antworte immer auf Deutsch.
""".strip()

def get_system_prompt() -> str:
    if IMPORTANT_RULES:
        return f"""{SYSTEM_PROMPT}

=== WICHTIGE REGELN (IMMER BEACHTEN) ===
{IMPORTANT_RULES}
=== ENDE DER WICHTIGEN REGELN ==="""
    return SYSTEM_PROMPT


def main():
    # TODO hier müsstet ihr nochmal schauen, ob das die von euch verwendete API ist
    USE_OPENAI_API = True

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", nargs="?", type=str, help="The query text.")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--iterations", type=int, default=1, 
                        help="Anzahl der Durchläufe (nur mit --input-file und --output-file)")
    parser.add_argument("--history", type=str, default="[]")
    args = parser.parse_args()

    if args.input_file:
        if not args.output_file:
            parser.error("--output-file ist erforderlich, wenn --input-file verwendet wird.")
        
        # Mehrfache Durchläufe
        if args.iterations > 1:
            process_prompt_file_multiple(args.input_file, args.output_file, USE_OPENAI_API, args.iterations)
        else:
            process_prompt_file(args.input_file, args.output_file, USE_OPENAI_API)
        return

    if not args.query_text:
        parser.error("query_text oder --input-file ist erforderlich.")

    try:
        chat_history = json.loads(args.history)
    except json.JSONDecodeError:
        chat_history = []

    query_rag(args.query_text, USE_OPENAI_API, chat_history=chat_history)


def query_rag(query_text: str, use_openai_api: bool = True, chat_history: list = None):
    if chat_history is None:
        chat_history = []
    
    # TODO das hier müsste auch noch angepasst werden
    embedding_platform = "lm-studio" if use_openai_api else "ollama"
    embedding_function = get_embedding_function(embedding_platform)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Hole relevante Dokumente
    docs = db.similarity_search(query_text, k=5)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Erstelle Model
    if use_openai_api: # TODO hier sind eigentlich die environment Variablen ausreichend, aber die defaults könntet ihr auch anpassen
        from langchain_openai import ChatOpenAI
        api_base = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
        model_name = os.getenv("LMSTUDIO_CHAT_MODEL", "qwen2.5-14b-instruct")
        api_key = os.getenv("API_KEY", "lm-studio")
        model = ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
        )
    else:
        model = ChatOllama(model="llama3.2:3b") # TODO der Fall ist für euch wahrscheinlich auch nicht relevant

    # Baue Nachrichten auf
    messages = [SystemMessage(content=get_system_prompt())]
    
    for entry in chat_history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    
    # Anfrage mit Kontext
    user_message = f"""Frage: {query_text}

Relevanter Kontext:
{context}

Beantworte die Frage. Nutze den Kontext NUR, wenn er zur Frage passt. Wenn der Kontext nicht relevant ist, ignoriere ihn."""
    
    messages.append(HumanMessage(content=user_message))

    # Abfrage
    response = model.invoke(messages)
    response_text = response.content if response.content else "No response from model."

    # Prüfe ob der Kontext in der Antwort verwendet wurde
    # Einfache Heuristik: Wenn Antwort sehr kurz und generisch, wurde Kontext wahrscheinlich nicht verwendet
    sources = []
    if _has_used_context(response_text, query_text):
        seen = set()
        for doc in docs:
            source_id = doc.metadata.get("id", None)
            if source_id and source_id not in seen:
                seen.add(source_id)
                sources.append(source_id)
    
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text, sources


def _has_used_context(response: str, query: str) -> bool:
    """Heuristik: Prüft ob der Kontext wahrscheinlich verwendet wurde."""
    # Wenn die Antwort sehr generisch/kurz ist, wurde wahrscheinlich kein Kontext verwendet
    generic_responses = [
        "wie geht es dir",
        "ich bin ein ernährungsberater",
        "ich bin ein chatbot",
        "wie kann ich dir helfen",
        "wer bist du",
        "was kannst du",
        "hallo",
        "guten tag",
        "es geht mir gut",
        "danke der nachfrage",
    ]
    
    # Wenn Frage sehr generisch ist
    query_lower = query.lower()
    for generic in generic_responses:
        if generic in query_lower:
            return False
    
    # Wenn Antwort zu kurz, war wahrscheinlich kein Kontext nötig
    if len(response) < 100:
        return False
    
    return True


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


def process_prompt_file(input_path: str, output_path: str, use_openai_api: bool = True):
    prompts = _load_prompts_from_file(input_path)
    if not prompts:
        raise ValueError(f"Keine Prompts in {input_path} gefunden.")

    with open(output_path, "w", encoding="utf-8") as handle:
        for prompt in prompts:
            start_time = time.time()
            response_text, sources = query_rag(prompt, use_openai_api)
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


def _get_output_filename(base_path: str, iteration: int) -> str:
    """Erstellt einen nummerierten Dateinamen."""
    # Splitte Pfad und Dateiname
    dir_path = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    
    # Trenne Name und Extension
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
        numbered_filename = f"{name}_{iteration}.{ext}"
    else:
        numbered_filename = f"{filename}_{iteration}"
    
    # Wenn kein Directory angegeben, nutze aktuelles Directory
    if not dir_path:
        return numbered_filename
    
    return os.path.join(dir_path, numbered_filename)


def process_prompt_file_multiple(input_path: str, output_path: str, use_openai_api: bool = True, iterations: int = 1):
    """Führt den Prompt-Durchlauf mehrfach aus und schreibt Output-Dateien sortiert nach Fragen."""
    prompts = _load_prompts_from_file(input_path)
    if not prompts:
        raise ValueError(f"Keine Prompts in {input_path} gefunden.")
    
    # Sammle alle Ergebnisse: {prompt: [(response, sources, duration, iteration), ...]}
    all_results = {prompt: [] for prompt in prompts}
    
    # Führe alle Durchläufe durch
    for iteration in range(1, iterations + 1):
        print(f"Durchlauf {iteration}/{iterations}...")
        for prompt in prompts:
            start_time = time.time()
            response_text, sources = query_rag(prompt, use_openai_api)
            duration = int(time.time() - start_time)
            pdf_sources = _extract_pdf_sources(sources)
            
            all_results[prompt].append((response_text, pdf_sources, duration, iteration))
        print(f"Durchlauf {iteration} abgeschlossen.\n")
    
    # Schreibe Output-Dateien: Eine Datei pro Frage
    for prompt_idx, prompt in enumerate(prompts, 1):
        numbered_output = _get_output_filename(output_path, prompt_idx)
        print(f"Schreibe Ergebnisse für Frage {prompt_idx} in {numbered_output}")
        
        with open(numbered_output, "w", encoding="utf-8") as handle:
            handle.write(f"Frage {prompt_idx}:\n")
            handle.write(f"{prompt}\n")
            handle.write("=" * 60 + "\n\n")
            
            # Schreibe alle Durchläufe für diese Frage
            for response_text, pdf_sources, duration, iteration in all_results[prompt]:
                handle.write(f"Durchlauf {iteration}:\n")
                handle.write("─" * 60 + "\n")
                handle.write(f"🤖 Assistant:\n")
                handle.write(f"{response_text}\n")
                if pdf_sources:
                    handle.write(f"\n(Quellen: {', '.join(pdf_sources)})\n")
                handle.write(f"Dauer: {duration}s\n")
                handle.write("─" * 60 + "\n\n")


if __name__ == "__main__":
    main()
