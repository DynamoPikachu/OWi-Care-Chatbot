from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedding_function(platform: str | None = None):
    normalized = (platform or "lm-studio").lower()
    if normalized in {"lm-studio", "lmstudio", "openai"}:
        return OpenAIEmbeddings(
            model="text-embedding-3-large",  # Name egal
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="lm-studio",
        )
    if normalized == "ollama":
        return OllamaEmbeddings(model="nomic-embed-text")
    raise ValueError(f"Unknown embedding platform: {platform}")
