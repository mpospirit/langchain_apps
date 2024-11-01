from typing import Union, List
from langchain_core.agents import AgentAction, AgentFinish
from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the input text by character count.
    """
    text = text.strip("'\n").strip('"')  # Removing the non alphanumeric characters
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]

    # Can also be imported from the hub (from langchain import hub)
    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        stop=[
            "\nObservation",
            "Observation",
        ],  # This tells LLM to stop generating text after the Observation token
        # We do this because we want LLM to keep generating text until it reaches the Observation token
        callbacks=[AgentCallbackHandler()],
    )
    intermidiate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of word mpospirit?",
                "agent_scratchpad": intermidiate_steps,
            }
        )

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermidiate_steps.append((agent_step, str(observation)))

        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of word mpospirit?",
                "agent_scratchpad": intermidiate_steps,
            }
        )

    if isinstance(agent_step, AgentFinish):
        print(f"Final answer: {agent_step.return_values}")
