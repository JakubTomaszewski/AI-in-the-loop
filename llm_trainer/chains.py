from enum import Enum
from typing import Callable, List

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec

from . import prompts
from .data_types import ClassInformation, Prompts, PromptStrategy
from .utils import class_info_to_xml


class ChainType(Enum):
    STRATEGY_GENERATION = "strategy_generation"
    STRATEGY_GENERATION_WITH_HISTORY = "strategy_generation_with_history"
    PROMPT_GENERATION = "prompt_generation"
    PROMPT_SUMMARIZATION = "prompt_summarization"


class ChainFactory:
    def __init__(self, llm):
        self.llm = llm

    def create_strategy_generation_with_history_chain(self, get_chat_history: Callable):
        strategy_generation_output_parser = PydanticOutputParser(
            pydantic_object=PromptStrategy
        )
        strategy_generation_prompt = ChatPromptTemplate(
            messages=[
                ("system", prompts.STRATEGY_GENRATION_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", prompts.STRATEGY_GENRATION_USER_PROMPT),
            ],
            input_variables=["class_information"],
            partial_variables={
                "output_format": strategy_generation_output_parser.get_format_instructions(),
            },
        )

        chain: Runnable[ClassInformation, PromptStrategy] = (
            {"class_information": RunnableLambda(lambda x: x["class_information"]), "chat_history": RunnableLambda(lambda x: x["chat_history"])}
            # | RunnableLambda(class_info_to_xml)
            | strategy_generation_prompt
            | self.llm
            | strategy_generation_output_parser
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_chat_history,
            input_messages_key="class_information",
            history_messages_key="chat_history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="session ID",
                    description="Unique identifier for the session.",
                    default="",
                    is_shared=True,
                )
            ],
        )
        return chain_with_history

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
            | RunnableLambda(class_info_to_xml)
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
            {"prompts": RunnableLambda(lambda x: x["prompts"])}
            | RunnableLambda(format_prompts)
            | prompt_summarization_prompt
            | self.llm
            | prompt_summarization_output_parser
        )
        return chain

    def create_chain(self, chain_type: ChainType, **kwargs) -> Runnable:
        if chain_type == ChainType.STRATEGY_GENERATION:
            return self.create_strategy_generation_chain(**kwargs)
        elif chain_type == ChainType.STRATEGY_GENERATION_WITH_HISTORY:
            return self.create_strategy_generation_with_history_chain(**kwargs)
        elif chain_type == ChainType.PROMPT_GENERATION:
            return self.create_prompt_generation_chain(**kwargs)
        elif chain_type == ChainType.PROMPT_SUMMARIZATION:
            return self.create_prompt_summarization_chain(**kwargs)
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
