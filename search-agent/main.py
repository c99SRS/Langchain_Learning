from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch



# @tool
# def search(query: str) -> str:
#     """Search for information about the given query and return a short answer."""
#     print(f"Searching for: {query}")
#     return  tavily.search(query=query)  #"Tokyo weather is sunny"


# Use a local Ollama model that supports tools (function calling)
# Make sure you have pulled it first: `ollama pull qwen3`
#llm = ChatOllama(model="gemma3:270m", temperature=0)


class Source(BaseModel):
    """ Schema for a source used by the agent """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources to generate answers")

llm = ChatGroq(model="openai/gpt-oss-120b")

# Instantiate the tool
tavily_tool = TavilySearch()  # you can pass options here if you like
tools = [tavily_tool]  #[search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

#print(llm.invoke("Hello!").content)
#llm = ChatOllama(model="qwen3", temperature=0)



def main():
    print("Hello, I am the agent...")
    #Passing below commented question gives error output, bcoz tool is trying to open another toold and it is failing
    # For the longer prompt (“…on LinkedIn. List their details”), the model decides it needs to:
    # Search, then “Open” individual job URLs to read details.
    # Since there is no open tool in your tools list, Groq validates the tool call and responds with a 400 error.
    #question="Search for 3 job posting for an AI engineer using langchain in the Bengaluru are on LInkedIn. List their details"
    question = (
    "search to find 3 job postings for an AI engineer using LangChain "
    "in the Bangalore area on LinkedIn."
    )
    #print("Question:", repr(question))
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job posting for an AI engineer using langchain in the Bengaluru are on LInkedIn")]})
    #result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(result)

    # `create_agent` returns a dict with a messages list; print only the final answer
    # messages = result.get("messages", [])
    # if messages:
    #     final_message = messages[-1]
    #     print(final_message.content)
    # else:
    #     # Fallback in case the structure is different
    #     print(result)


if __name__ == "__main__":
    main()
