import os
import json
import argparse
import subprocess

from time import sleep
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run all classes in a dataset")
    parser.add_argument("--metadata", type=str, help="Metadata file for the dataset", default="./data/pods/metadata.json")
    parser.add_argument("--template", type=str, help="Job file template")
    parser.add_argument("--output", type=str, help="Output directory for job files")
    parser.add_argument(
        "--wait_time", type=int, help="Wait time between job submissions", default=300
    )
    parser.add_argument(
        "--class_performance_output_path",
        type=str,
        help="Output file for class performance",
    )
    parser.add_argument(
        "--results_output",
        type=str,
        help="Output directory for results.json files",
        default="/scratch-shared/jtomaszewski/personalized_reps/evaluation_output",
    )
    parser.add_argument(
        "--num_synthetic_samples",
        type=int,
        help="Number of synthetic samples to generate for each class",
        default=50,
    )
    parser.add_argument(
        "--num_triplets",
        type=int,
        help="Number of triplets to generate for each class",
        default=500,
    )

    parser.add_argument(
        "--log_file",
        type=str,
        help="Log file for the script",
        default="generate_data_all_classes_parallel_PROMPTS.log",
    )
    
    parser.add_argument(
        "--negatives_path",
        type=str,
        help="Path to the negatives",
    )

    return parser.parse_args()


def check_job_status(job_ids):
    while True:
        running_jobs = subprocess.run(["squeue"], capture_output=True, text=True)
        if not any(job_id in running_jobs.stdout for job_id in job_ids):
            break
        sleep(30)  # Wait before checking again


def submit_job(job_file):
    result = subprocess.run(
        ["sbatch", output_job_file_path], capture_output=True, text=True
    )

    job_id = result.stdout.strip().split()[-1]  # Extract job ID
    return job_id


if __name__ == "__main__":
    args = parse_args()

    # read metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)["class_to_sc"]

    logger.add(
        f"{args.log_file}_evaluate_all_classes_parallel.log",
        format="{time} {level} {message}",
        level="INFO",
    )

    logger.info("Running for classes: ", metadata.keys())

    os.makedirs(args.output, exist_ok=True)

    logger.info("Creating evaluation job files for all classes in the dataset")

    job_ids = []

    for class_name, class_category in metadata.items():
        logger.info(f"Generating job file for class: {class_name}")

        with open(args.template, "r") as f:
            job_file = f.read()

        # generate a job file for each class
        output_job_file_path = os.path.join(args.output, f"{class_name}.job")
        with open(output_job_file_path, "w") as f:
            job_file = job_file.replace("${CLASS_NAME}", class_name)
            job_file = job_file.replace("${OUTPUT_PATH}", args.results_output)
            job_file = job_file.replace("${DATASET_METADATA}", args.metadata)
            job_file = job_file.replace("${NEGATIVES_PATH}", args.negatives_path)
            job_file = job_file.replace("${NUM_SYNTHETIC_SAMPLES}", str(args.num_synthetic_samples))
            job_file = job_file.replace("${NUM_TRIPLETS}", str(args.num_triplets))
            f.write(job_file)

        # Run sbatch output_job_file_path
        logger.info(f"Submitting job file for class: {class_name}")

        job_id = submit_job(output_job_file_path)
        logger.info(f"Job ID: {job_id}")

        job_ids.append(job_id)

        # Wait until the job is finished
        sleep(200)
        check_job_status([job_id])
        
        # If the slurm output does not exist or contains an error or has been cancelled, retry
        slurm_output_file = f"slurm_output_{job_id}.out"
        if not os.path.exists(slurm_output_file) or "State: ERROR" in open(slurm_output_file).read() or "State: CANCELLED" in open(slurm_output_file).read():
            logger.info(f"Job {job_id} failed or was cancelled. Retrying...")
            job_id = submit_job(output_job_file_path)
            logger.info(f"Retrying job ID: {job_id}")
            job_ids.append(job_id)
            
            # Wait until the job is finished
            sleep(200)
            check_job_status([job_id])

    logger.info("All job files have been submitted. Waiting for all jobs to complete.")
    check_job_status(job_ids)
    logger.info("All SLURM jobs have completed.")

    output_results_combined = {}

    for class_name in metadata.keys():
        results_file = os.path.join(
            args.results_output, class_name, "results.json"
        )

        with open(results_file, "r") as f:
            result = json.load(f)
            classification_score = result["custom_lora_dinov2_vitb14"][class_name][
                "avg_classification"
            ]
            logger.info(f"{class_name}: {classification_score}")

            output_results_combined[class_name] = classification_score

    logger.info(f"Writing combined results to {args.class_performance_output_path}")

    with open(args.class_performance_output_path, "w") as f:
        json.dump(output_results_combined, f)

    logger.info("Done Evaluating all classes.")
