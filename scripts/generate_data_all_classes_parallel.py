import os
import json
import argparse
import subprocess

from time import sleep
from loguru import logger

logger.add(
    "generate_data_all_classes_parallel.log",
    format="{time} {level} {message}",
    level="INFO",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all classes in a dataset")

    parser.add_argument(
        "--class_sample_count_path",
        type=str,
        help="File containing the number of samples for each class",
    )
    parser.add_argument("--metadata", type=str, help="Metadata file for the dataset")
    parser.add_argument("--template", type=str, help="Job file template")
    parser.add_argument("--output", type=str, help="Output directory for job files")
    parser.add_argument(
        "--wait_time", type=int, help="Wait time between job submissions", default=120
    )
    parser.add_argument(
        "--generate_data_output_path",
        type=str,
        help="Output path for generated data",
        default="/scratch-shared/jtomaszewski/personalized_reps/synthetic_data/pods",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        help="Path to the prompts file",
        default="./configs/prompts/gpt_prompts_pods.json",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to generate for each class",
        default=100,
    )

    return parser.parse_args()


def check_job_status(job_ids):
    while True:
        running_jobs = subprocess.run(["squeue"], capture_output=True, text=True)
        if not any(job_id in running_jobs.stdout for job_id in job_ids):
            break
        sleep(30)  # Wait before checking again


if __name__ == "__main__":
    args = parse_args()

    # read metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)["class_to_sc"]

    with open(args.class_sample_count_path, "r") as f:
        class_sample_count = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    logger.info("Creating data generation job files for all classes in the dataset")

    job_ids = []

    for class_name, class_category in metadata.items():
        logger.info(f"Generating job file for class: {class_name}")

        with open(args.template, "r") as f:
            job_file = f.read()

        # generate a job file for each class
        output_job_file_path = os.path.join(args.output, f"{class_name}.job")
        with open(output_job_file_path, "w") as f:
            job_file = job_file.replace("${CLASS_NAME}", class_name)
            job_file = job_file.replace("${CLASS_CATEGORY}", class_category)
            job_file = job_file.replace(
                "${NUM_SAMPLES}", str(class_sample_count[class_name])
            )
            job_file = job_file.replace(
                "${OUTPUT_PATH}", str(args.generate_data_output_path)
            )
            job_file = job_file.replace("${PROMPTS_PATH}", str(args.prompts_path))
            job_file = job_file.replace("${NUM_SAMPLES}", str(args.num_samples))

            f.write(job_file)

        # Run sbatch output_job_file_path
        logger.info(f"Submitting job file for class: {class_name}")
        # os.system(f"sbatch {output_job_file_path}")

        result = subprocess.run(
            ["sbatch", output_job_file_path], capture_output=True, text=True
        )
        job_id = result.stdout.strip().split()[-1]  # Extract job ID
        job_ids.append(job_id)

        sleep(args.wait_time)

    logger.info("All job files have been submitted. Waiting for all jobs to complete.")
    check_job_status(job_ids)
    logger.info("All SLURM jobs have completed.")
