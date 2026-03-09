import os
from typing import Iterable

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class LMStudioEmbeddings(OpenAIEmbeddings):
    def _tokenize(
        self, texts: list[str], chunk_size: int
    ) -> tuple[Iterable[int], list[str], list[int], list[int]]:
        tokens: list[str] = []
        indices: list[int] = []
        token_counts: list[int] = []

        max_chars = self.embedding_ctx_length
        for i, text in enumerate(texts):
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            for j in range(0, len(text), max_chars):
                chunk_text = text[j : j + max_chars]
                if not chunk_text:
                    continue
                tokens.append(chunk_text)
                indices.append(i)
                token_counts.append(len(chunk_text))

        return range(0, len(tokens), chunk_size), tokens, indices, token_counts


def get_embedding_function(platform: str | None = None):
    # TODO auch hier müsst ihr nochmal drüberschauen, ob die Schnittstelle euch taugt
    normalized = (platform or "lm-studio").lower()
    if normalized in {"lm-studio", "lmstudio", "openai"}:
        api_base = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
        model = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "text-embedding-3-small")
        max_chars = int(os.getenv("LMSTUDIO_EMBEDDING_CHUNK_CHARS", "2000"))
        return LMStudioEmbeddings(
            model=model,
            openai_api_base=api_base,
            openai_api_key="lm-studio",
            embedding_ctx_length=max_chars,
        )
    if normalized == "ollama":
        return OllamaEmbeddings(model="nomic-embed-text")
    raise ValueError(f"Unknown embedding platform: {platform}")
