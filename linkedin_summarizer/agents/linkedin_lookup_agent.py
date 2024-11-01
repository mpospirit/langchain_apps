import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from tools.tools import get_linkedin_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    template = """
    Given the full name {full_name} of a person from I want you to get a link to their LinkedIn profile.
    Your answer should be a URL to the LinkedIn profile.
    And you answer should start with "https://www.linkedin.com/in/".
    """

    prompt_template = PromptTemplate(input_variables=["full_name"], template=template)

    tools_for_agent = [
        Tool(
            name="Crawl Google for LinkedIn Profile Page",
            func=get_linkedin_profile_url,
            description="This tool crawls Google for LinkedIn profile page of a person.",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        prompt = react_prompt,
        llm=llm,
        tools=tools_for_agent,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(input={"input": prompt_template.format_prompt(full_name=name)})

    linked_profile_url = result["output"]

    return linked_profile_url


if __name__ == "__main__":
    print(lookup())