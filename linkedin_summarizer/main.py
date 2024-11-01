import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup

load_dotenv()  # This reads the key-value pair from .env file and adds them to environment variable

def get_information(name: str):
    linkedin_username = linkedin_lookup(name)
    linkedin_data = scrape_linkedin_profile(linkedin_username)

    summary_template = """
    Given the information {information} about a person from I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(
        temperature=0, # This sets the creativity of the AI, 0 means it won't be creative
        model_name="gpt-4o-mini",
    )

    chain = summary_prompt_template | llm | StrOutputParser()

    result = chain.invoke(input={"information": linkedin_data})

    return result


if __name__ == "__main__":
    print(get_information("Çağrı Gökpunar"))