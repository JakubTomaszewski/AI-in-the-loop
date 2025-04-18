import os
import json
import argparse
import subprocess

from time import sleep
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run all classes in a dataset")

    parser.add_argument("--metadata", type=str, help="Metadata file for the dataset")
    parser.add_argument("--template", type=str, help="Job file template")
    parser.add_argument("--output", type=str, help="Output directory for job files")
    parser.add_argument(
        "--wait_time", type=int, help="Wait time between job submissions", default=None
    )
    parser.add_argument(
        "--generate_data_output_path",
        type=str,
        help="Output path for generated data",
        # default="/scratch-shared/jtomaszewski/personalized_reps/synthetic_data/pods",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        help="Path to the prompts file",
        # default="./configs/prompts/gpt_prompts_pods.json",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to generate for each class",
        default=50,
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Log file for the script",
        default="generate_data_all_classes_parallel_PROMPTS.log",
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
    
    logger.info(f"Running script with arguments: {args}")

    if args.wait_time is None:
        args.wait_time = args.num_samples * 2

    logger.add(
        f"{args.log_file}_generate_data_all_classes_parallel.log",
        format="{time} {level} {message}",
        level="INFO",
    )

    # read metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)["class_to_sc"]

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
                "${OUTPUT_PATH}", str(args.generate_data_output_path)
            )
            job_file = job_file.replace("${PROMPTS_PATH}", str(args.prompts_path))
            job_file = job_file.replace("${NUM_SAMPLES}", str(args.num_samples))

            f.write(job_file)

        logger.info(
            f"Saved job file for class: {class_name} to path: {output_job_file_path}"
        )

        # Run sbatch output_job_file_path
        logger.info(f"Submitting job file for class: {class_name}")

        result = subprocess.run(
            ["sbatch", output_job_file_path], capture_output=True, text=True
        )
        job_id = result.stdout.strip().split()[-1]  # Extract job ID
        job_ids.append(job_id)

        logger.info(f"Job ID: {job_id}")

        # Wait until the job is finished
        sleep(60)
        check_job_status([job_id])

    logger.info("All job files have been submitted. Waiting for all jobs to complete.")
    check_job_status(job_ids)
    logger.info("All SLURM jobs have completed.")
