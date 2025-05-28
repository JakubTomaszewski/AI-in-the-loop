import pandas as pd
import os
import random
from util.data_util import get_paths


def sample_triplet(triplet):
    """
    Randomize positive/negative ordering.

    Args:
        triplet (tuple): A tuple containing reference, positive, and negative samples.

    Returns:
        tuple: A tuple containing the randomized triplet and the label indicating the position of the positive sample.
    """
    ref, pos, neg = triplet

    # Randomly rearrange the positive and negative
    if random.randint(1, 2) == 1:
        return (ref, pos, neg), 0
    else:
        return (ref, neg, pos), 1


def get_positives(synthetic_paths, real_paths, ref_type):
    """
    Get positive samples based on the reference type.

    Args:
        synthetic_paths (list): List of synthetic paths.
        real_paths (list): List of real paths.
        ref_type (str): Type of reference ('real', 'synthetic', or 'both').

    Returns:
        list: A list containing the reference and positive samples.
    """
    if ref_type == "real":
        ref_paths = real_paths
    elif ref_type == "synthetic":
        ref_paths = synthetic_paths
    elif ref_type == "both":
        ref_paths = real_paths + synthetic_paths

    ref = random.sample(ref_paths, 1)[0]
    pos = random.sample(synthetic_paths, 1)[0]
    positives = [ref, pos]
    return positives


def get_negative(negative_paths):
    """
    Sample a negative sample from a list containing negative samples
    """
    return random.sample(negative_paths, 1)


def setup_triplets(
    cls,
    cls_name,
    positives_root,
    num_synthetic,
    negatives_root,
    real_train_root,
    num_triplets,
    save_path,
    ref_type="real",
    verbose=False,
):
    """
    Sets up triplets for a given class by generating a dataset of triplets consisting of
    reference, positive, and negative images. The triplets will be used for training
    contrastive learning models.

    Args:
        cls (int): Class id to make the triplet dataset for.
        cls_name (str): Corresponding class name.
        positives_root (str): Root directory of positive samples.
        num_synthetic (int): Number of synthetic positives to use.
        negatives_root (str): Root directory of negative samples.
        real_train_root (str): Root directory of real training data.
        num_triplets (int): Number of triplets to generate per class.
        save_path (str): Path to save the generated dataset CSV.
        ref_type (str, optional): Type of anchor image in each triplet, either "real" or "synthetic". Defaults to "real".
    """

    print(f"\nRunning on class {cls}: {cls_name}")
    print(f"Using {num_synthetic} synthetic positives to make {num_triplets} triplets.")
    print("Saving to ", save_path)

    real_train_paths = os.path.join(real_train_root, cls_name)
    positive_train_paths = os.path.join(positives_root, cls_name)
    negative_train_paths = os.path.join(negatives_root, cls_name)

    print(f"Getting real from {real_train_paths}")
    print(f"Getting positives from {positive_train_paths}")
    print(f"Getting negatives from {negative_train_paths}")

    # Get sample paths
    real_train_paths = get_paths(real_train_paths)
    positive_train_paths = get_paths(positive_train_paths)
    negative_train_paths = get_paths(negative_train_paths)

    # Sample triplets
    random.shuffle(real_train_paths)
    random.shuffle(positive_train_paths)
    random.shuffle(negative_train_paths)

    print(f"{len(real_train_paths)} real anchors")
    print(f"{len(positive_train_paths)} synthetic positives")
    print(f"{len(negative_train_paths)} synthetic negatives")

    csv_dict = []
    train_triplets = []

    print("Making train triplets")
    progress = 0
    while len(train_triplets) < num_triplets:
        negative = get_negative(negative_train_paths)
        positives = get_positives(positive_train_paths, real_train_paths, ref_type)
        train_triplets.append(positives + negative)

        new_progress = int(len(train_triplets) * 100 // num_triplets)
        if new_progress % 10 == 0 and new_progress != progress and verbose:
            progress = new_progress
            print(f"{progress} % generated")

    for i in range(num_triplets):
        (ref, left, right), label = sample_triplet(train_triplets[i])
        csv_dict.append(
            {
                "id": i,
                "label": label,
                "ref_path": ref,
                "left_path": left,
                "right_path": right,
            }
        )

    csv = pd.concat([pd.DataFrame(csv_dict)])
    csv.to_csv(save_path, index=False)
    print("done :)")
