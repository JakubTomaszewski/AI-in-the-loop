import os
import json
import argparse

from string import Template
from time import sleep


def parse_args():
    parser = argparse.ArgumentParser(description='Run all classes in a dataset')
    parser.add_argument('--metadata', type=str, help='Metadata file for the dataset')
    parser.add_argument('--template', type=str, help='Job file template')
    parser.add_argument('--output', type=str, help='Output directory for job files')
    parser.add_argument('--wait_time', type=int, help='Wait time between job submissions', default=5)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # read metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)["class_to_sc"]

    os.makedirs(args.output, exist_ok=True)

    for class_name, class_category in metadata.items():
        print("Generating job file for class: ", class_name)
        
        with open(args.template, 'r') as f:
            job_file = f.read()

        # generate a job file for each class
        output_job_file_path = os.path.join(args.output, f'{class_name}.job')
        with open(output_job_file_path, 'w') as f:
            job_file = job_file.replace("${CLASS_NAME}", class_name)
            job_file = job_file.replace("${CLASS_CATEGORY}", class_category)
            f.write(job_file)
        
        # Run sbatch output_job_file_path
        print(f"Submitting job file for class: {class_name}")
        os.system(f"sbatch {output_job_file_path}")
        sleep(args.wait_time)
