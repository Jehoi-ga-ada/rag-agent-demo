import os
from typing import Any, Dict

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain.

    Args:
        query (str): The user's question
    """
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)
    
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get("source", "Unknown")}\n\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query using retrieved documents

    Args:
        query (str): The user's question

    Returns:
        Dict[str, Any]: Dictionary containing 
            - answer: The generated answer
            - context: List of retrieved documents
    """
    
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    messages = [{"role":"user", "content":query}]
    
    response = agent.invoke({"messages": messages})
    
    answer = response["messages"][-1].content
    
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }
    
if __name__=="__main__":
    result = run_llm("what are deep agents?")
    print(result)