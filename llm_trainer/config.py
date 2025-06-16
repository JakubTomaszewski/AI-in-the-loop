import argparse


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
        default=50,
        help="Initial number of synthetic samples to generate for each class.",
    )

    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        # default="/home/jtomaszewski/personalized-rep/data/pods/metadata_subset.json",
        default="/home/jtomaszewski/personalized-rep/data/dogs/metadata.json",
        help="Path to the dataset metadata file.",
    )

    parser.add_argument(
        "--generation_template_file",
        type=str,
        default="/home/jtomaszewski/personalized-rep/jobs/generate_data/generate_data_lora_sd3_template.job",
        # default="/home/jtomaszewski/personalized-rep/jobs/generate_data/generate_data_template.job",
        help="Path to the template file to use for generating data.",
    )

    parser.add_argument(
        "--evaluation_template_file",
        type=str,
        default="/home/jtomaszewski/personalized-rep/jobs/evaluation/evaluate_template.job",
        help="Path to the template file to use for evaluating data.",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="o3-mini",
        # default="gpt-4.1",
        help="LLM model to use for generation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for the model",
    )

    parser.add_argument(
        "--generate_data_output_path",
        type=str,
        help="Output path for generated data",
        # default="/scratch-shared/jtomaszewski/personalized_reps/synthetic_data/pods/sd3_lora/",
        default="/scratch-shared/jtomaszewski/personalized_reps/synthetic_data/dogs/sd3_lora/",
    )

    parser.add_argument(
        "--generated_prompts_path",
        type=str,
        help="Output path for generated prompts",
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/PROMPTS/{timestamp}/iteration_{iteration_number}/prompts.json",
    )

    parser.add_argument(
        "--evaluation_output_path",
        type=str,
        help="Output path for evaluation results",
        default="/scratch-shared/jtomaszewski/personalized_reps/evaluation_output/",
    )

    parser.add_argument(
        "--prompt_summaries_path",
        type=str,
        help="Output path for prompt summaries",
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/PROMPTS/{timestamp}/iteration_{iteration_number}/prompt_summaries.json",
    )

    parser.add_argument(
        "--prompt_generation_strategy_path",
        type=str,
        help="Path to the prompt generation strategy file",
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/PROMPTS/{timestamp}/iteration_{iteration_number}/prompt_generation_strategies.json",
    )

    parser.add_argument(
        "--class_performance_output_path",
        type=str,
        default="/home/jtomaszewski/personalized-rep/llm_trainer_metadata/PROMPTS/{timestamp}/iteration_{iteration_number}/class_performance.json",
        help="Path to the output class performance file.",
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

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["pods", "dogs", "df2"],
        default="dogs",
        help="Name of the dataset to use.",
    )

    parser.add_argument(
        "--real_data_path",
        type=str,
        # default='./data/pods',
        default="./data/dogs",
        help="Path to the real data directory.",
    )

    parser.add_argument(
        "--negatives_path",
        type=str,
        # default="/scratch-shared/jtomaszewski/personalized_reps/pods_negatives/",
        default="/scratch-shared/jtomaszewski/personalized_reps/dogs_negatives/",
        help="Path to the negatives root directory.",
    )

    parser.add_argument(
        "--append_generated_data",
        action="store_true",
        help="Append generated data to existing data.",
    )

    parser.add_argument(
        "--without_prompt_strategies",
        action="store_true",
        help="If set, the prompt strategies will not be used in the generation process.",
    )

    return parser.parse_args()
