from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedding_function(platform):
    match platform:
        """case "bedrock"
        embeddings = BedrockEmbeddings(
            credentials_profile_name="default", region_name="us-east-1"
        )"""
        case "ollama":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        case "lm-stuio":
        embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",  # Name egal
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm-studio",
        )
    return embeddings
