import json
import argparse
import asyncio

from loguru import logger
from pydantic import BaseModel

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


from dotenv import load_dotenv

load_dotenv()


GENERATION_PROMPT = """
<instructions>
Generate a list of {num_prompts} prompts for a class category: {class_category}. 
Those prompts will be used to generate synthetic data for the class, hence they should be descriptive, diverse, and representative of the class category.

You MUST generate exactly {num_prompts} prompts for the class category.

Ensure that each prompt follows the format below:
</instructions>

<example>
Input: class_category="bottle", num_prompts=3
Output: ["A <new1> bottle on a picnic table", "A <new1> bottle on a picnic table", "A <new1> bottle on a picnic table"]
</example>

<output_format>
{format_instructions}
</output_format>
"""


class OutputPrompts(BaseModel):
    prompts: list[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all classes in a dataset")

    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        default="/home/jtomaszewski/personalized-rep/data/pods/metadata.json",
        help="Path to the dataset metadata file.",
    )

    parser.add_argument(
        "--num_prompts",
        type=int,
        help="Number of prompts to generate for each class.",
        default=100,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file to save the generated prompts.",
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/generated_prompts.json",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    output_parser = PydanticOutputParser(pydantic_object=OutputPrompts)
    prompt = PromptTemplate(
        template=GENERATION_PROMPT,
        input_variables=["class_category", "num_prompts"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    llm = ChatOpenAI(model="gpt-4o")

    chain = prompt | llm | output_parser

    with open(args.dataset_metadata_path, "r") as f:
        metadata = json.load(f)["class_to_sc"]

    generated_prompts = {}

    for class_name, class_category in metadata.items():
        logger.info("Generating prompts for class: ", class_name)

        response = await chain.ainvoke(
            {"class_category": class_category, "num_prompts": args.num_prompts}
        )

        generated_prompts[class_name] = response.prompts

        assert len(generated_prompts[class_name]) == args.num_prompts

    # tasks = []
    # for class_name, class_category in metadata.items():
    #     logger.info("Generating prompts for class: ", class_name)
    #     tasks.append(
    #         chain.ainvoke(
    #             {"class_category": class_category, "num_prompts": args.num_prompts}
    #         )
    #     )

    # responses = await asyncio.gather(*tasks)

    # for class_name, response in zip(metadata.keys(), responses):
    #     generated_prompts[class_name] = response.prompts
    #     assert len(generated_prompts[class_name]) == args.num_prompts

    with open(args.output_file, "w") as f:
        json.dump(generated_prompts, f)

    logger.info(f"Generated prompts saved to: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())


# Example: "A <new1> bottle on a picnic table",
