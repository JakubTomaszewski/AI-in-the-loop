#!/usr/bin/env python3
# filepath: clip_score_calculator.py

import os
import argparse
import json
import torch
from PIL import Image
import clip
import re
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate CLIP scores for images across directories"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the main directory containing class subdirectories",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run CLIP on (cuda/cpu)",
    )
    parser.add_argument(
        "--output", type=str, default="clip_scores.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to process per class",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use (e.g., ViT-B/32, ViT-L/14)",
    )
    return parser.parse_args()


def clean_prompt(filename):
    # Remove number prefix and file extension
    # Pattern: NUMBER_prompt.EXTENSION
    prompt = re.sub(r"^\d+_(.+)\.[^\.]+$", r"\1", filename)

    # Remove "sks" or "<new1>" keywords
    prompt = prompt.replace("sks", "").replace("<new1>", "").strip()

    # Clean up any extra spaces
    prompt = re.sub(r"\s+", " ", prompt).strip()

    return prompt


def calculate_clip_score(image, text, model, preprocess, device):
    """Calculate CLIP score between an image and text prompt"""
    try:
        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize([text]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).item()

        return similarity
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def main():
    args = parse_args()

    # Load CLIP model
    device = args.device
    model, preprocess = clip.load(args.clip_model, device=device)

    # Get sorted list of directories (classes)
    class_dirs = sorted(os.listdir(args.path))

    clip_scores_by_class = {}
    all_scores = []
    
    with open("sd_15_clip_scores.json", "r") as f:
        sd_15_clip_scores = json.load(f)
    print(f"Loaded {len(sd_15_clip_scores)} CLIP scores from sd_15_clip_scores.json")

    for dir_name in tqdm(class_dirs, desc="Processing directories"):
        print(f"Processing class: {dir_name}")
        class_dir = os.path.join(args.path, dir_name)
        
        if dir_name not in sd_15_clip_scores["class_scores"]:
            print(f"Skipping class {dir_name} as it is not in sd_15_clip_scores.json")
            continue

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Get image files in the directory
        image_files = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ][: args.num_images]

        # Process up to num_images from this directory
        scores = []

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)

            # Extract the prompt from the filename
            prompt = clean_prompt(img_file)

            try:
                # Load the image
                image = Image.open(img_path).convert("RGB")

                # Calculate CLIP score
                score = calculate_clip_score(image, prompt, model, preprocess, device)

                if score is not None:
                    scores.append(score)
                    all_scores.append(score)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Store the average score for this class
        if scores:
            clip_scores_by_class[dir_name] = {
                "individual_scores": scores,
                "average_score": sum(scores) / len(scores),
            }

    # Calculate the overall average
    overall_average = sum(all_scores) / len(all_scores) if all_scores else 0

    # Prepare the results dictionary
    results = {"class_scores": clip_scores_by_class, "overall_average": overall_average}

    # Save the results to a JSON file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"CLIP scores saved to {args.output}")
    print(f"Overall average CLIP score: {overall_average:.4f}")


if __name__ == "__main__":
    main()
