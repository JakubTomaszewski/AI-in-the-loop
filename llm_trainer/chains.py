from enum import Enum
from llm_trainer import prompts

from typing import List
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
    CommaSeparatedListOutputParser,
    ListOutputParser
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI

from llm_trainer.data_types import ClassDetails, ClassInformation, PromptStrategy, Prompts


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
        strategy_generation_prompt = PromptTemplate(
            template=prompts.STRATEGY_GENRATION_PROMPT,
            input_variables=["class_information"],
            partial_variables={
                "output_format": strategy_generation_output_parser.get_format_instructions()
            },
        )
        chain: Runnable[ClassInformation, PromptStrategy] = (
            {"class_information": RunnablePassthrough()}
            | strategy_generation_prompt
            | self.llm
            | strategy_generation_output_parser
        )
        return chain

    def create_prompt_generation_chain(self):
        prompt_generation_output_parser = PydanticOutputParser(
            pydantic_object=Prompts
        )
        prompt_generation_prompt = PromptTemplate(
            template=prompts.PROMPT_GENERATION_PROMPT,
            input_variables=["prompt_strategy", "num_prompts", "object_category"],
            partial_variables={
                "output_format": prompt_generation_output_parser.get_format_instructions()
            },
        )
        chain: Runnable[PromptStrategy, List[str]] = (
            {
                "prompt_strategy": RunnablePassthrough(),
                "num_prompts": RunnablePassthrough(),
                "object_category": RunnablePassthrough(),
            }
            | prompt_generation_prompt
            | self.llm
            | prompt_generation_output_parser
        )
        return chain

    def create_prompt_summarization_chain(self):
        def format_prompts(prompts):
            formatted_prompts = "\n,".join(f"'{prompt}'" for prompt in prompts["prompts"]["prompts"])
            return "[\n" + formatted_prompts + "\n]"

        prompt_summarization_output_parser = StrOutputParser()
        prompt_summarization_prompt = PromptTemplate(
            template=prompts.PROMPT_SUMMARIZATION_PROMPT,
            input_variables=["prompts"],  # List of prompts
        )
        chain: Runnable[List[str], str] = (
            {"prompts": RunnablePassthrough()}  # TODO: Debug this as the prompts are not passed correctly
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
