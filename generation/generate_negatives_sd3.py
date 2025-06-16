from diffusers import StableDiffusion3Pipeline

import os
import argparse
import json
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        # default="configs/prompts/gpt_negative_prompts_pods.json",
        default="configs/prompts/gpt_negative_prompts_dogs.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # default="/scratch-shared/jtomaszewski/personalized_reps/pods_negatives/",
        default="/scratch-shared/jtomaszewski/personalized_reps/dogs_negatives/",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    if os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists. Exiting.")
        exit()

    os.makedirs(args.output_dir)

    with open(args.prompts_file) as file:
        prompts_metadata = json.load(file)

    for obj, prompts in prompts_metadata.items():
        print(f"Generating negatives for object: {obj}")
        
        class_path = os.path.join(args.output_dir, f"{obj}_negatives")
        os.makedirs(class_path, exist_ok=True)

        for prompt in prompts:
            print(f"Generating image for prompt: {prompt}")

            image = pipe(prompt, num_inference_steps=25).images[0]

            image.save(os.path.join(class_path, prompt.replace(" ", "_") + ".png"))
