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
    real_data_path: str,
    synthetic_data_path: str,
    negatives_path: str,
    evaluation_output_path: str,
    num_synthetic_samples: int,
    dataset_name: str,
):
    evaluation_script = f"""python scripts/evaluate_all_classes_parallel.py \
        --metadata {dataset_metadata_path} \
        --template {template_file} \
        --output evaluate_scripts \
        --class_performance_output_path {class_performance_output_path} \
        --real_data_path {real_data_path} \
        --synthetic_data_path {synthetic_data_path} \
        --negatives_path {negatives_path} \
        --num_synthetic_samples {num_synthetic_samples} \
        --num_triplets {num_synthetic_samples * 10} \
        --results_output {evaluation_output_path} \
        --dataset_name {dataset_name} \
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
        --num_samples {args.num_synthetic_samples} \
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

    llm = ChatOpenAI(model=args.llm_model, temperature=args.temperature)
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
            logger.info(
                f"Removing the existing evaluation output path and creating a new one: {args.evaluation_output_path}"
            )
            shutil.rmtree(args.evaluation_output_path)

        if os.path.exists(args.evaluation_output_path):
            logger.info(
                f"Removing the existing evaluation output file: {args.evaluation_output_path} and creating a new one"
            )
            os.remove(args.evaluation_output_path)

        os.makedirs(args.evaluation_output_path, exist_ok=True)

        logger.info("Generating data using the generated prompts")

        generate_data(
            prompts_path=generated_prompts_path,
            dataset_metadata_path=args.dataset_metadata_path,
            template_file=args.generation_template_file,
            generate_data_output_path=generate_data_output_path,
        )

        logger.info(f"Generated data using the provided prompts.")

        class_performance_output_path = args.class_performance_output_path.format(
            timestamp=current_time, iteration_number=iteration_number
        )

        ##### 3. Train & Evaluate the model performance #####
        # Remove cache and embeddings
        if os.path.exists(args.cache_dir) and os.path.isdir(args.cache_dir):
            logger.info(
                f"Removing the existing cache directory and creating a new one: {args.cache_dir}"
            )
            shutil.rmtree(args.cache_dir)

        if os.path.exists(args.cache_dir):
            logger.info(
                f"Removing the existing cache file: {args.cache_dir} and creating a new one"
            )
            os.remove(args.cache_dir)
        os.makedirs(args.cache_dir, exist_ok=True)

        if os.path.exists(args.embeddings_dir):
            logger.info(
                f"Removing the existing embeddings directory and creating a new one: {args.embeddings_dir}"
            )
            shutil.rmtree(args.embeddings_dir)
        if os.path.exists(args.embeddings_dir):
            logger.info(
                f"Removing the existing embeddings file: {args.embeddings_dir} and creating a new one"
            )
            os.remove(args.embeddings_dir)

        os.makedirs(args.embeddings_dir, exist_ok=True)

        num_synthetic_samples = (
            args.num_synthetic_samples * (iteration_number + 1)
            if args.append_generated_data
            else args.num_synthetic_samples
        )

        class_performance = evaluate_class_performance(
            class_performance_output_path=class_performance_output_path,
            dataset_metadata_path=args.dataset_metadata_path,
            template_file=args.evaluation_template_file,
            real_data_path=args.real_data_path,
            synthetic_data_path=generate_data_output_path,
            negatives_path=args.negatives_path,
            evaluation_output_path=args.evaluation_output_path,
            num_synthetic_samples=num_synthetic_samples,
            dataset_name=args.dataset_name,
        )

        logger.info(
            f"Model performance at iteration {iteration_number}: {class_performance}"
        )

        iteration_number += 1
