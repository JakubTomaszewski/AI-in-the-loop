"""
A LLM-based training which controls an end-to-end training pipeline. This includes:
1. Keeping track of the training performance history (metrics per class)
2. Assessing the model performance using metrics provided by the training tool
3. Generating more data to improve performance of some specific classes. This includes:
    a. Controlling the number of synthetic samples for each class
    b. Controlling the prompt used for generating the images
"""

import os
import json
import shutil
import argparse
import subprocess
import numpy as np

from typing import Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI


TRAINING_PROMPT = """
You are a data scientist controlling the training process of a machine learning model, which is trained using AI-generated synthetic data. The model is trained to classify images into different classes.
You are provided with a history of the model's performance on different classes, measured by some metric (typically accuracy i.e. the higher the better).

Metric history:
<metric_history>
{metric_history}
</metric_history>

<instructions>
You can control the number of synthetic samples generated for each class. For instance, if a specific class is underperforming, you can generate more synthetic samples for that class to potentially improve the model's performance.
It is your responsibility to decide how many synthetic samples to generate for each class.

The model can be retrained many times, and you are responsible for monitoring its performance on each class to ensure that it is learning effectively.

Using the metric history, decide how many synthetic samples to generate for each class in order to improve the model's performance.
If the performance for a particular class decreases after generating more synthetic samples, you can revert the changes by generating fewer synthetic samples for that class.
Start by generating a few synthetic samples for each class and monitor the model's performance. If the model's performance is not satisfactory, you can generate more synthetic samples for specific classes.

Note: You can generate a maximum of 100 synthetic samples for each class. Thus, start with a small number of synthetic samples and increase the number as needed.

Do not any write code. Output only the number of synthetic samples to generate for each class in the format specified below.
</instructions>

<output_format>
{format_instructions}
</output_format>
"""


class Metrics(BaseModel):
    iteration: int = Field(..., description="The iteration number")
    metrics: dict[str, Any] = Field(..., description="The performance metrics")


class ImageGenerationParams(BaseModel):
    class_samples: dict[str, int] = Field(
        ..., description="The number of synthetic samples to generate for each class"
    )


class LoggingCallback(BaseCallbackHandler):
    def on_chain_end(self, outputs):
        logger.info(f"Chain output: {outputs}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_loop_iterations",
        type=int,
        default=5,
        help="Maximum number of loop iterations.",
    )

    parser.add_argument(
        "--num_synthetic_samples",
        type=int,
        default=10,
        help="Initial number of synthetic samples to generate for each class.",
    )

    parser.add_argument(
        "--class_sample_count_path",
        type=str,
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/iteration_{iteration_number}/class_sample_count.json",
        help="Path to the output class sample count file.",
    )

    parser.add_argument(
        "--class_performance_output_path",
        type=str,
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/iteration_{iteration_number}/class_performance.json",
        help="Path to the output class performance file.",
    )

    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        default="/home/jtomaszewski/personalized-rep/data/pods/metadata.json",
        help="Path to the dataset metadata file.",
    )

    parser.add_argument(
        "--generation_template_file",
        type=str,
        default="/home/jtomaszewski/personalized-rep/jobs/generate_data/generate_data_template.job",
        help="Path to the template file to use for generating data.",
    )

    parser.add_argument(
        "--evaluation_template_file",
        type=str,
        default="/home/jtomaszewski/personalized-rep/jobs/evaluation/evaluate_template.job",
        help="Path to the template file to use for evaluating data.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the model",
    )

    parser.add_argument(
        "--generate_data_output_path",
        type=str,
        help="Output path for generated data",
        default="/scratch-shared/jtomaszewski/personalized_reps/synthetic_data/pods",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch-shared/jtomaszewski/personalized_reps/cache/",
        help="Path to the cache directory.",
    )

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/scratch-shared/jtomaszewski/personalized_reps/embeddings/",
        help="Path to the embeddings directory.",
    )

    return parser.parse_args()


def evaluate_class_performance(class_performance_output_path: str, template_file: str):
    evaluation_script = f"""python scripts/evaluate_all_classes_parallel.py \
        --metadata /home/jtomaszewski/personalized-rep/data/pods/metadata.json \
        --template {template_file} \
        --output evaluate_scripts \
        --class_performance_output_path {class_performance_output_path}
    """

    logger.info(f"Running evaluation script: {evaluation_script}")
    result = subprocess.run(
        evaluation_script, shell=True, capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)

    logger.info(
        f"Completed evaluation script. Loading class performance from {class_performance_output_path}"
    )

    with open(class_performance_output_path, "r") as f:
        class_performance = json.load(f)

    logger.info(f"Class performance: {class_performance}")
    logger.info(
        f"Overall model performance: {np.mean(list(class_performance.values()))}"
    )

    return class_performance


def generate_data(
    class_sample_count_path: str,
    dataset_metadata_path: str,
    template_file: str,
    generate_data_output_path: str,
):
    """
    Generate synthetic samples for each class.

    Args:
        class_samples: A dictionary with the class name as the key and the number of samples as the value.
    """
    # Call the generate_data_all_classes_parallel script and pass the class_samples as an argument
    generation_script = f"""python scripts/generate_data_all_classes_parallel.py \
        --class_sample_count_path {class_sample_count_path} \
        --metadata {dataset_metadata_path} \
        --template {template_file} \
        --output generate_data_scripts \
        --generate_data_output_path {generate_data_output_path}
        """

    logger.info(f"Running generation script: {generation_script}")
    subprocess.run(generation_script, shell=True)


def format_history(metric_history):
    print("Metric history in format_history: ", metric_history)

    return "\n".join(str(metric.model_dump()) for metric in metric_history)


if __name__ == "__main__":
    args = parse_args()

    with open(args.dataset_metadata_path) as file:
        metadata = json.load(file)
        class_names = metadata["classes"]

    metric_history = []

    output_parser = PydanticOutputParser(pydantic_object=ImageGenerationParams)
    prompt = PromptTemplate(
        template=TRAINING_PROMPT,
        input_variables=["metric_history"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=args.temperature)
    # llm = ChatOpenAI(model="gpt-4o", temperature=args.temperature)
    llm = ChatOpenAI(model="o1-mini")

    chain = (
        {"metric_history": RunnablePassthrough() | format_history}
        | prompt
        | llm
        | output_parser
    )

    # Generate 10 synthetic samples for each class
    logger.info(
        f"Generating initial synthetic samples for each class ({args.num_synthetic_samples})"
    )

    class_sample_count = ImageGenerationParams(
        class_samples={
            class_name: args.num_synthetic_samples for class_name in class_names
        }
    )
    
    iteration_number = 0
    class_sample_count_path = args.class_sample_count_path.format(
        iteration_number=iteration_number
    )
    os.makedirs(os.path.dirname(class_sample_count_path), exist_ok=True)

    with open(class_sample_count_path, "w") as f:
        json.dump(class_sample_count.class_samples, f)

    while iteration_number < args.max_loop_iterations:
        # 1. Generate data
        logger.info(f"Generating data for iteration {iteration_number}")

        # Cleanup the existing AI-generated data
        logger.info(
            f"Removing existing AI-generated data from path: {args.generate_data_output_path}"
        )

        shutil.rmtree(args.generate_data_output_path, ignore_errors=True)
        shutil.rmtree(
            "/scratch-shared/jtomaszewski/personalized_reps/evaluation_output/",
            ignore_errors=True,
        )

        class_sample_count_path = args.class_sample_count_path.format(
            iteration_number=iteration_number
        )

        generate_data(
            class_sample_count_path,
            args.dataset_metadata_path,
            args.generation_template_file,
            args.generate_data_output_path,
        )

        # 2. Evaluate model performance
        # Remove cache and embeddings
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        shutil.rmtree(args.embeddings_dir, ignore_errors=True)

        class_performance_output_path = args.class_performance_output_path.format(
            iteration_number=iteration_number
        )
        os.makedirs(os.path.dirname(class_performance_output_path), exist_ok=True)

        # Remove the current class performance file
        shutil.rmtree(class_performance_output_path, ignore_errors=True)

        model_results = evaluate_class_performance(
            class_performance_output_path, args.evaluation_template_file
        )

        logger.info(
            f"Model performance at iteration {iteration_number}: {model_results}"
        )

        # add the class performance & number of synthetic samples to the training history (format it as a json)
        metric_history.append(
            Metrics(iteration=iteration_number, metrics=model_results)
        )

        # 3. Get the number of synthetic samples to generate for each
        logger.info(
            f"Getting the number of synthetic samples to generate for iteration {iteration_number}"
        )
        class_sample_count = chain.invoke(metric_history)
        logger.info(f"Data to generate: {class_sample_count}")

        # Save the synthetic sample count to a file
        with open(class_sample_count_path, "w") as f:
            json.dump(class_sample_count.class_samples, f)

        iteration_number += 1
