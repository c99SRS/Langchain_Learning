from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily=TavilyClient()

@tool
def search(query: str) -> str:
    """Search for information about the given query and return a short answer."""
    print(f"Searching for: {query}")
    return  tavily.search(query=query)  #"Tokyo weather is sunny"


# Use a local Ollama model that supports tools (function calling)
# Make sure you have pulled it first: `ollama pull qwen3`
#llm = ChatOllama(model="gemma3:270m", temperature=0)


llm = ChatGroq(model="openai/gpt-oss-120b")
tools = [search]
agent = create_agent(model=llm, tools=tools)

#print(llm.invoke("Hello!").content)

#llm = ChatOllama(model="qwen3", temperature=0)



def main():
    print("Hello, I am the agent...")
    question = "What is the weather in Tokyo?"
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # `create_agent` returns a dict with a messages list; print only the final answer
    messages = result.get("messages", [])
    if messages:
        final_message = messages[-1]
        print(final_message.content)
    else:
        # Fallback in case the structure is different
        print(result)


if __name__ == "__main__":
    main()
