import argparse

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

AGENT_SYSTEM_PROMPT = """
You are a helpful assistant. Use the `search_docs` tool to find relevant context
from the vector store before answering. If needed, call the tool multiple times.
Answer only from retrieved context. If the context is insufficient, say so.
""".strip()


def main():
    # set parameters
    USE_LM_STUDIO = True

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text, USE_LM_STUDIO)


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
        model = ChatOpenAI(
            model="lmstudio",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="lm-studio",
        )
    else:
        model = ChatOllama(model="llama3.2:3b")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(model, [search_docs], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[search_docs], verbose=False)

    ### hier geschieht die Magie
    response = agent_executor.invoke({"input": query_text})
    ####################################

    response_text = response["output"]

    sources = []
    seen = set()
    for doc in retrieved_docs:
        source_id = doc.metadata.get("id", None)
        if source_id and source_id not in seen:
            seen.add(source_id)
            sources.append(source_id)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
