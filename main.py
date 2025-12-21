from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    """Schema for a source used by an agent """
    url: str = Field(description="Source URL")

class AgentResponse(BaseModel):
    """Schema for an agent response with answer and source"""
    answer: str = Field(description="The agent's final answer to the quer")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent to answer the question")

llm = ChatOpenAI()
tools = [TavilySearch]
agent  = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="Search for three AI Engineer jobs in Munich, Germany on linkedin and list three postings.")})
    print(result)


if __name__ == "__main__":
    main()
