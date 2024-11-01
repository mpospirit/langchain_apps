# This is a risky project since the following code let's the agent execute Python code

from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentType, Tool
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    create_python_agent,
)

load_dotenv()


def main():
    instructions = """
    You are an agent designed to write and execute python code to answer questions.
    You have access to a Python REPL, which you can use to execute code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running the code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as your answer.
    """

    # ReAct Agent prompt
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        prefix="""You are an agent designed to answer questions about a CSV file using Python pandas.
        You MUST use the python_repl_ast tool to execute any pandas commands.
        When you need to run code, use the Action: python_repl_ast format.
        Example correct format:
        Thought: I need to analyze the data
        Action: python_repl_ast
        Action Input: df.groupby('Season')['EpisodeNo'].count()
        """,
    )

    # Router Grand Agent
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str: any]:
        return python_agent_executor.invoke({"input": original_prompt})


    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""
                Useful when you need to transform natural language to python and execute the python code.
                Returning results of the code execution.
                DOES NOT ACCEPT CODE AS INPUT.
                """,
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""
                Useful when you need to answer questions about the episode_info.csv file.
            Input should be a natural language question about the data.
            The agent will handle the necessary pandas operations internally.
                """,
        ),
    ]

    prompt = base_prompt.partial(instructions="""You are a helpful assistant that routes questions to the appropriate specialized agent.
    For Python code execution, use the Python Agent.
    For questions about the episode_info.csv file, use the CSV Agent.
    Always use the most appropriate agent for the task.""")

    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
    )

    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {
                "input": "Which season has the highest number of episodes?",
            }
        )
    )

    print(
        grand_agent_executor.invoke(
            {
                "input": "What is 4 + 4?",
            }
        )
    )


if __name__ == "__main__":
    main()
