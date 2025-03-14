from langchain_openai.chat_models import ChatOpenAI
from chains import ChainFactory, ChainType
from data_types import ClassDetails, ClassInformation, PromptStrategy, PromptSummaries


class_information = ClassInformation(
    class_details={
        "class1": ClassDetails(performance=0.5, prompt_summary="A summary of the prompts for class1"),
        "class2": ClassDetails(performance=0.8, prompt_summary="A summary of the prompts for class2"),
        "class3": ClassDetails(performance=0.3, prompt_summary="A summary of the prompts for class3"),
    }
)


prompt_strategy = PromptStrategy(
    class_strategies={
        "class1": "Generate prompts that are more detailed and specific to the class. For example, 'A high resolution image of a <new1> {{object_category}}' + details about the object or scene.",
        "class2": "Add different object orientations to the prompts. For example, 'A bright image of a <new1> {{object_category}}' + details about the object orientation or perspective.",
        "class3": "Generate prompts that have different, forest-like backgrounds. For example, 'A <new1> {{object_category}}' + details about the forest-like background.",
    }
)


prompts = [
    "A high resolution image of a <new1> {{object_category}}",
    "A bright image of a <new1> {{object_category}}",
    "A <new1> {{object_category}}",
    "A picture containing a <new1> {{object_category}} on a grassy field",
]



if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o")
    chain_factory = ChainFactory(llm)

    # # Strategy generation
    # strategy_generation_chain = chain_factory.create_chain(
    #     ChainType.STRATEGY_GENERATION
    # )
    # # print(strategy_generation_prompt.format(class_information=class_information))
    # print(strategy_generation_chain.invoke({"prompt_strategy": prompt_strategy}))

    
    # prompt_generation_chain = chain_factory.create_chain(ChainType.PROMPT_GENERATION)
    # # print(prompt_generation_prompt.format(prompt_strategy=prompt_strategy))
    # print(prompt_generation_chain.invoke({"prompt_strategy": prompt_strategy}))
    
    print("Creating chain")
    prompt_summarization_chain = chain_factory.create_chain(ChainType.PROMPT_SUMMARIZATION)
    print("Chain created")
    # print(prompt_summarization_prompt.format(prompts=prompts))
    print(prompt_summarization_chain.invoke({"prompts": prompts}))
    