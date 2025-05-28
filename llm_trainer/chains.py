from enum import Enum
from llm_trainer import prompts

from typing import List
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

from llm_trainer.data_types import ClassInformation, PromptStrategy, Prompts


class ChainType(Enum):
    STRATEGY_GENERATION = "strategy_generation"
    PROMPT_GENERATION = "prompt_generation"
    PROMPT_SUMMARIZATION = "prompt_summarization"


class ChainFactory:
    def __init__(self, llm):
        self.llm = llm

    def create_strategy_generation_chain(self):
        strategy_generation_output_parser = PydanticOutputParser(
            pydantic_object=PromptStrategy
        )
        strategy_generation_prompt = ChatPromptTemplate(
            messages=[
                ("system", prompts.STRATEGY_GENRATION_PROMPT),
                ("human", prompts.STRATEGY_GENRATION_USER_PROMPT),
            ],
            input_variables=["class_information"],
            partial_variables={
                "output_format": strategy_generation_output_parser.get_format_instructions()
            },
        )
        chain: Runnable[ClassInformation, PromptStrategy] = (
            {"class_information": RunnableLambda(lambda x: x["class_information"])}
            | strategy_generation_prompt
            | self.llm
            | strategy_generation_output_parser
        )
        return chain

    def create_prompt_generation_chain(self):
        prompt_generation_output_parser = PydanticOutputParser(pydantic_object=Prompts)
        prompt_generation_prompt = ChatPromptTemplate(
            messages=[
                ("system", prompts.PROMPT_GENERATION_PROMPT),
                ("human", prompts.PROMPT_GENERATION_USER_PROMPT),
            ],
            input_variables=["prompt_strategy", "num_prompts", "object_category"],
            partial_variables={
                "output_format": prompt_generation_output_parser.get_format_instructions()
            },
        )
        chain: Runnable[PromptStrategy, List[str]] = (
            {
                "prompt_strategy": RunnableLambda(lambda x: x["prompt_strategy"]),
                "num_prompts": RunnableLambda(lambda x: x["num_prompts"]),
                "object_category": RunnableLambda(lambda x: x["object_category"]),
            }
            | prompt_generation_prompt
            | self.llm
            | prompt_generation_output_parser
        )
        return chain

    def create_prompt_summarization_chain(self):
        def format_prompts(prompts):
            formatted_prompts = "\n".join(
                f"'{prompt}'," for prompt in prompts["prompts"]
            )
            return "[\n" + formatted_prompts + "\n]"

        prompt_summarization_output_parser = StrOutputParser()
        prompt_summarization_prompt = ChatPromptTemplate(
            messages=[
                ("system", prompts.PROMPT_SUMMARIZATION_PROMPT),
                ("human", prompts.PROMPT_SUMMARIZATION_USER_PROMPT),
            ],
            input_variables=["prompts"],  # List of prompts
        )
        chain: Runnable[List[str], str] = (
        {
            "prompts": RunnableLambda(lambda x: x["prompts"]) }
            | RunnableLambda(format_prompts)
            | prompt_summarization_prompt
            | self.llm
            | prompt_summarization_output_parser
        )
        return chain

    def create_chain(self, chain_type: ChainType):
        if chain_type == ChainType.STRATEGY_GENERATION:
            return self.create_strategy_generation_chain()
        elif chain_type == ChainType.PROMPT_GENERATION:
            return self.create_prompt_generation_chain()
        elif chain_type == ChainType.PROMPT_SUMMARIZATION:
            return self.create_prompt_summarization_chain()
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
