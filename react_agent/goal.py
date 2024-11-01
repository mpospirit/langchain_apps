from tabnanny import verbose
from langchain.agents import initialize_agent, AgentType, tool, AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the input text by character count.
    """
    text = text.strip("'\n").strip('"')
    return len(text)

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent_executor: AgentExecutor = initialize_agent(
        tools=[get_text_length],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    agent_executor.invoke({"input": "What is the length of word mpospirit?"})