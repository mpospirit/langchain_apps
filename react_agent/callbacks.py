from typing import Any
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# What we are doing here is overriding some of the methods in the BaseCallbackHandler class
class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any
    ) -> Any:
        print(f"Prompt to LLM was: \n{prompts[0]}")
        print("=====================================")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> Any:
        print(f"LLM response: \n{response.generations[0][0].text}")
        print("=====================================")
        
