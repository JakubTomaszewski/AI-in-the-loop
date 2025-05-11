import os
import json
import sys
import argparse
import subprocess

from time import sleep

from loguru import logger
from datetime import datetime


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file = os.path.join(os.getcwd(), f"train_dreambooth_{current_time}.log")
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


def check_job_status(job_ids):
    while True:
        running_jobs = subprocess.run(["squeue"], capture_output=True, text=True)
        if not any(job_id in running_jobs.stdout for job_id in job_ids):
            break
        sleep(60)  # Wait before checking again


def parse_args():
    parser = argparse.ArgumentParser(description="Run all classes in a dataset")
    parser.add_argument("--metadata", type=str, help="Metadata file for the dataset")
    parser.add_argument("--template", type=str, help="Job file template")
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for job files",
        default="train_dreambooth_sd3_scripts",
    )
    # parser.add_argument('--wait_time', type=int, help='Wait time between job submissions', default=1800)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # read metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)["class_to_sc"]

    os.makedirs(args.output, exist_ok=True)

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
            f.write(job_file)

        # Run sbatch output_job_file_path
        logger.info(f"Submitting job file for class: {class_name}")

        result = subprocess.run(
            ["sbatch", output_job_file_path], capture_output=True, text=True
        )
        job_id = result.stdout.strip().split()[-1]  # Extract job ID
        job_ids.append(job_id)

        logger.info(f"Job ID: {job_id}")

        # Wait until the job is finished
        check_job_status([job_id])

    logger.info("All job files have been submitted. Waiting for all jobs to complete.")
    check_job_status(job_ids)
    logger.info("All SLURM jobs have completed.")
