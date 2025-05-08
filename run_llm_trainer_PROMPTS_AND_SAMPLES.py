"""
A LLM-based training which controls an end-to-end training pipeline. Three stage LLM reasoning which includes:
* per-class: summarize prompts in terms of background / geometry / scene composition etc.
* given all the summaries and the performances, ask the LLM to generate a high level strategy how new images should be generated for each class in one call
* per-class: turn this high-level strategy into hundreds of prompts
"""
import re
import sys
import os
import asyncio
import shutil
import json
import subprocess
import backoff
import numpy as np

from collections import defaultdict
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
from datetime import datetime

from llm_trainer import (
    ChainFactory,
    ChainType,
    ClassInformation,
    ClassDetails,
    PromptStrategy,
    parse_args,
    ClassPerformanceAndSamplesInformation,
    NumSyntheticSamples,
)


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file = os.path.join(os.getcwd(), f"llm_trainer_{current_time}.log")
log_level = "INFO"

log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
logger.add(
    sys.stderr,
    level=log_level,
    format=log_format,
    colorize=True,
    backtrace=True,
    diagnose=True,
)
logger.add(
    sys.stdout,
    level=log_level,
    format=log_format,
    colorize=False,
    backtrace=True,
    diagnose=True,
)
logger.add(
    log_file,
    level=log_level,
    format=log_format,
    colorize=False,
    backtrace=True,
    diagnose=True,
)


def evaluate_class_performance(
    class_performance_output_path: str,
    dataset_metadata_path: str,
    template_file: str,
    synthetic_data_path: str,
    negatives_path: str,
    evaluation_output_path: str,
    num_synthetic_samples: int,
):
    evaluation_script = f"""python scripts/evaluate_all_classes_parallel.py \
        --metadata {dataset_metadata_path} \
        --template {template_file} \
        --output evaluate_scripts \
        --class_performance_output_path {class_performance_output_path} \
        --synthetic_data_path {synthetic_data_path} \
        --negatives_path {negatives_path} \
        --num_synthetic_samples {num_synthetic_samples} \  # TODO: This has to be passed as a file ...
        --num_triplets {num_synthetic_samples * 10} \
        --results_output {evaluation_output_path} \
        --log_file {log_file}
    """

    logger.info(f"Running evaluation script: {evaluation_script}")
    result = subprocess.run(
        evaluation_script, shell=True, capture_output=True, text=True
    )
    # subprocess.run(evaluation_script, shell=True)

    logger.info(result.stdout)
    logger.info(result.stderr)

    logger.info(
        f"Completed evaluation script. Loading class performance from {class_performance_output_path}"
    )

    with open(class_performance_output_path, "r") as f:
        class_performance = json.load(f)

    overall_performance = np.mean(list(class_performance.values()))

    logger.info(f"Class performance: {class_performance}")
    logger.info(f"Overall model performance: {overall_performance}")
    return class_performance


def generate_data(
    prompts_path: str,
    dataset_metadata_path: str,
    template_file: str,
    generate_data_output_path: str,
    num_samples: int
):
    """
    Generate synthetic samples for each class.

    Args:
        class_samples: A dictionary with the class name as the key and the number of samples as the value.
    """

    generation_script = f"""python scripts/generate_data_all_classes_parallel_PROMPTS.py \
        --metadata {dataset_metadata_path} \
        --template {template_file} \
        --output generate_data_scripts \
        --generate_data_output_path {generate_data_output_path} \
        --prompts_path {prompts_path} \
        --num_samples {num_samples} \
        --log_file {log_file}
        """

    logger.info(f"Running generation script: {generation_script}")
    subprocess.run(generation_script, shell=True)


async def generate_prompts_all_classes(
    class_names, prompt_generation_strategy: dict
) -> dict[str, list[str]]:
    @backoff.on_exception(backoff.expo, Exception, max_time=300, max_tries=5)
    async def invoke_chain(class_name: str) -> str:
        await asyncio.sleep(5)
        prompts = await prompt_generation_chain.ainvoke(
            {
                "prompt_strategy": prompt_generation_strategy[class_name],
                "num_prompts": args.num_synthetic_samples,
                "object_category": object_categories[class_name],
            }
        )
        prompts = [parse_prompt(prompt) for prompt in prompts.prompts]

        print(f"Num prompts: {len(prompts)} for class: {class_name}")
        # assert len(prompts) == args.num_synthetic_samples  # TODO: Add a mechanism for checking whether the number of prompts is correct/close to the expected number
        return prompts

    tasks = [invoke_chain(class_name) for class_name in class_names]
    results = await asyncio.gather(*tasks)
    return {class_name: result for class_name, result in zip(class_names, results)}


async def summarize_all_classes(class_names, generated_prompts: dict) -> dict[str, str]:
    @backoff.on_exception(backoff.expo, Exception, max_time=300, max_tries=5)
    async def invoke_chain(class_name: str) -> str:
        await asyncio.sleep(5)
        return await prompt_summarization_chain.ainvoke(
            {"prompts": generated_prompts[class_name]}
        )

    tasks = [invoke_chain(class_name) for class_name in class_names]
    results = await asyncio.gather(*tasks)
    return {class_name: result for class_name, result in zip(class_names, results)}


# TODO: Potentially refactor the functions above to be one function as below
# async def run_async(chain: Runnable, class_names, **kwargs) -> dict[str, str]:
#     @backoff.on_exception(backoff.expo, Exception, max_time=300, max_tries=5)
#     async def invoke_chain(class_name: str) -> str:
#         return await chain.ainvoke(**kwargs)

#     tasks = [invoke_chain(class_name) for class_name in class_names]
#     results = await asyncio.gather(*tasks)
#     return dict(zip(class_names, results))


# async def summarize_all_classes(class_names) -> dict[str, str]:
#     return await run_async(
#         prompt_summarization_chain,
#         class_names,
#         prompts=lambda class_name: generated_prompts[class_name],
#     )


# async def generate_prompts_all_classes(class_names) -> dict[str, str]:
#     return await run_async(
#         prompt_generation_chain, class_names, prompt_strategy=prompt_generation_strategy
#     )


def parse_prompt(prompt_str):
    """
    Removes the 'Prompt NUMBER:' prefix from the prompt string.
    Example:
    Input: "Prompt 1: A close-up image of a sks mug on a table, with a cup of coffee next to it, and a book in the background"
    Output: "A close-up image of a sks mug on a table, with a cup of coffee next to it, and a book in the background"
    """
    return re.sub(r"^Prompt\s+\d+:\s+", "", prompt_str)


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {args}")

    with open(args.dataset_metadata_path) as file:
        metadata = json.load(file)

        class_names = metadata["classes"]
        object_categories = metadata["class_to_sc"]

    logger.info(f"Running LLM trainer for {len(class_names)} classes: {class_names}")

    llm = ChatOpenAI(model="o3-mini")
    chain_factory = ChainFactory(llm)

    # Strategy generation
    strategy_generation_chain = chain_factory.create_chain(
        ChainType.STRATEGY_GENERATION
    )

    # Prompt generation
    prompt_generation_chain = chain_factory.create_chain(ChainType.PROMPT_GENERATION)

    # Prompt summarization
    prompt_summarization_chain = chain_factory.create_chain(
        ChainType.PROMPT_SUMMARIZATION
    )
    
    num_synthetic_samples_generation_chain = chain_factory.create_chain(ChainType.NUM_SYNTHETIC_SAMPLES)

    # Initially - no strategy
    prompt_generation_strategy = PromptStrategy(
        class_strategies=defaultdict(
            lambda: """Generate extremely simple and short prompts randomly for each class.
            The prompts should say that the object is in some place or scenario that is relevant to the object.
            
            For example:
            "A sks mug on a stool",
            "A sks mug on a table",
            "A sks mug in a sink full of dishes",
            "A sks mug on a picnic blanket",
            
            Do not overcomplicate the prompts. Keep them super simple, concise, and relevant to the object.
            """
        )
    )

    class_performance_history: list[ClassPerformanceAndSamplesInformation] = []
    num_synthetic_samples: dict[str, int] = defaultdict(
        lambda: args.num_synthetic_samples
    )  # Default number of samples for each class

    iteration_number = 0
    
    while iteration_number < args.max_loop_iterations:
        logger.info(f"Starting iteration: {iteration_number}")

        ##### 1. Generate prompts using the strategy #####
        logger.info("Generating prompts using the strategies")

        generated_prompts = asyncio.run(
            generate_prompts_all_classes(class_names, prompt_generation_strategy)
        )

        logger.info(f"Prompts generated")

        # Save the generated prompts to a file
        generated_prompts_path = args.generated_prompts_path.format(
            timestamp=current_time, iteration_number=iteration_number
        )
        os.makedirs(os.path.dirname(generated_prompts_path), exist_ok=True)
        with open(generated_prompts_path, "w") as f:
            f.write(json.dumps(generated_prompts))

        logger.info(f"Generated prompts saved to: {args.generated_prompts_path}")

        ##### 2. Generate data using the generated prompts #####
        generate_data_output_path = os.path.join(
            args.generate_data_output_path,
            current_time,
            f"iteration_{iteration_number}",
        )

        # Copy the existing generations from the previous iteration
        if args.append_generated_data and iteration_number > 0:
            previous_iteration_path = os.path.join(
                args.generate_data_output_path,
                current_time,
                f"iteration_{iteration_number - 1}",
            )
            shutil.copytree(previous_iteration_path, generate_data_output_path)

        logger.info(f"Path for generated data: {generate_data_output_path}")

        if os.path.exists(args.evaluation_output_path) and os.path.isdir(
            args.evaluation_output_path
        ):
            shutil.rmtree(args.evaluation_output_path)

        if os.path.exists(args.evaluation_output_path):
            subprocess.run(f"rm {args.evaluation_output_path}")

        if iteration_number == 0:
            # Initial number of samples
            num_synthetic_samples = args.num_synthetic_samples
        else:
            num_synthetic_samples = ... # TODO

        logger.info("Generating data using the generated prompts")

        generate_data(
            prompts_path=generated_prompts_path,
            dataset_metadata_path=args.dataset_metadata_path,
            template_file=args.generation_template_file,
            generate_data_output_path=generate_data_output_path,
            num_samples=num_synthetic_samples  # TODO: This has to be passed as a file ...
        )

        logger.info(f"Generated data using the provided prompts.")

        class_performance_output_path = args.class_performance_output_path.format(
            timestamp=current_time, iteration_number=iteration_number
        )

        ##### 3. Train & Evaluate the model performance #####
        # Remove cache and embeddings
        if os.path.exists(args.cache_dir):
            shutil.rmtree(args.cache_dir)

        if os.path.exists(args.embeddings_dir):
            shutil.rmtree(args.embeddings_dir)

        # num_synthetic_samples = (
        #         args.num_synthetic_samples * (iteration_number + 1)
        #         if args.append_generated_data
        #         else args.num_synthetic_samples
        #     )

        class_performance = evaluate_class_performance(
            class_performance_output_path,
            args.dataset_metadata_path,
            args.evaluation_template_file,
            generate_data_output_path,
            args.negatives_path,
            args.evaluation_output_path,
            num_synthetic_samples,  # TODO: This has to be passed as a file ...
        )
        
        class_performance_history.append(
            ClassPerformanceAndSamplesInformation(
                iteration=iteration_number,
                class_accuracy=class_performance,
                num_synthetic_samples=num_synthetic_samples,
            )
        )

        logger.info(
            f"Model performance at iteration {iteration_number}: {class_performance}"
        )

        ##### 4. Summarize the prompts used for generation (for each class) #####
        logger.info("Summarizing the prompts used for generation")

        prompt_summaries = asyncio.run(
            summarize_all_classes(class_names, generated_prompts)
        )

        logger.info("Prompt summaries generated")

        # Save the prompt summaries to a file
        prompt_summaries_path = args.prompt_summaries_path.format(
            timestamp=current_time, iteration_number=iteration_number
        )
        os.makedirs(os.path.dirname(prompt_summaries_path), exist_ok=True)
        with open(prompt_summaries_path, "w") as f:
            f.write(json.dumps(prompt_summaries))

        ##### 5. Generate prompt strategy #####
        class_information = ClassInformation(
            class_details={
                class_name: ClassDetails(
                    performance=class_performance[class_name],
                    prompt_summary=prompt_summaries[class_name],
                )
                for class_name in class_names
            }
        )

        prompt_generation_strategy = strategy_generation_chain.invoke(
            {
                "class_information": class_information,
            }
        )

        logger.info("Prompt generation strategy generated")

        # Save the prompt generation strategy to a file
        prompt_generation_strategy_path = args.prompt_generation_strategy_path.format(
            timestamp=current_time, iteration_number=iteration_number
        )
        os.makedirs(os.path.dirname(prompt_generation_strategy_path), exist_ok=True)
        with open(prompt_generation_strategy_path, "w") as f:
            f.write(json.dumps(prompt_generation_strategy.class_strategies))

        ##### 6. Generate the number of synthetic samples for each class #####
        num_synthetic_samples = num_synthetic_samples_generation_chain.invoke(
            {
                "class_information": class_performance_history,
            }
        )

        iteration_number += 1
