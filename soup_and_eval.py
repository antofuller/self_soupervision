import os
import torch
from torch.utils.data import DataLoader
import argparse

import data
import utils
import models


def soup_and_eval(model_paths, dataset_name, gpu_id):
    # Load all state dicts
    state_dicts = []
    for path in model_paths:
        sd = torch.load(path, weights_only=True, map_location="cpu")
        state_dicts.append(sd)
        print(f"Loaded: {path}")

    # Uniform soup
    n = len(state_dicts)
    weights = [1.0 / n] * n
    souped_sd = utils.make_weighted_params(state_dicts, weights, gpu_id)

    # Build model from souped state dict
    num_classes = souped_sd["head.weight"].shape[0]
    encoder_params = {k.replace("encoder.", ""): v for k, v in souped_sd.items() if k.startswith("encoder.")}
    head_params = {k.replace("head.", ""): v for k, v in souped_sd.items() if k.startswith("head.")}
    model = models.ViTClassifier(encoder_params, head_params, num_classes).to(gpu_id)

    # Prepare test data
    test_dataset = data.miniVTAB(
        dataset_name=dataset_name,
        split="test",
        augmentation="none",
        do_two_views=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    # Evaluate
    acc = utils.test_model(model, test_loader, gpu_id)
    print(f"Soup accuracy on {dataset_name}: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    dataset_names = ["cifar100", "eurosat"]

    for dataset_name in dataset_names:
        ft_dir = f"{args.models_dir}/fine_tuned_vtab/{dataset_name}"

        # Collect all fine-tuned checkpoints
        model_paths = []
        for root, dirs, files in os.walk(ft_dir):
            for f in files:
                if f.endswith(".pt"):
                    model_paths.append(os.path.join(root, f))

        if not model_paths:
            print(f"No fine-tuned models found for {dataset_name} in {ft_dir}")
            continue

        print(f"\nSouping {len(model_paths)} models for {dataset_name}")
        acc = soup_and_eval(model_paths, dataset_name, args.gpu_id)

        utils.append_to_csv(
            f"{args.save_dir}/soup_and_eval_results.csv",
            [dataset_name, model_paths, acc]
        )
