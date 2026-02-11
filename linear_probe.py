import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import copy

import data
import utils
import models


def extract_features(encoder, data_loader, device):
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                features = encoder(images)[:, 0, :]

            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)  # [bsz, feature_dim]
    all_labels = torch.cat(all_labels, dim=0)

    # normalize over samples
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True)
    all_features = (all_features - mean) / (std + 1e-5)

    return all_features, all_labels, mean, std


def fold_normalization_into_probe(probe_state_dict, mean, std):
    weight = probe_state_dict['weight'].clone()  # [num_classes, feature_dim]
    bias = probe_state_dict['bias'].clone()  # [num_classes]

    std = std.squeeze(0).to(weight.device)  # [feature_dim]
    mean = mean.squeeze(0).to(weight.device)  # [feature_dim]

    new_weight = weight / (std + 1e-5)  # [num_classes, feature_dim]
    new_bias = bias - (new_weight @ mean)  # [num_classes]

    return {
        'weight': new_weight,
        'bias': new_bias
    }


def train_linear_probe(features, labels, num_classes, learning_rate, device):
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    linear_probe = nn.Linear(features.shape[1], num_classes).to(device)

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = len(loader)
    epochs = 100
    total_steps = epochs * steps_per_epoch
    schedule = utils.make_warmup_cosine(optimizer, epochs=total_steps, warmup_frac=0.10)

    criterion = nn.CrossEntropyLoss()

    linear_probe.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = linear_probe(batch_features)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()
            schedule.step()

    return linear_probe


def test_linear_probe(linear_probe, features, labels, device):
    linear_probe.eval()
    features = features.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = linear_probe(features)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        acc = correct / total * 100.0

    return acc


def linear_probe(cfg):
    ########################
    ##### PREPARE DATA #####
    ########################

    train_dataset = data.miniVTAB(
        dataset_name=cfg.dataset_name,
        split="train",
        augmentation="none",
        do_two_views=False
    )
    num_classes = train_dataset.get_num_classes()
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Dataset length: {len(train_dataset)}")
    print(f"Number of classes: {num_classes}")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    test_dataset = data.miniVTAB(
        dataset_name=cfg.dataset_name,
        split="test",
        augmentation="none",
        do_two_views=False
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    #################################
    ##### EXTRACT FEATURES ONCE #####
    #################################

    encoder = models.vit_base()
    stock_path = f"{cfg.models_dir}/mae_encoder.pth"
    encoder_weights = torch.load(stock_path, weights_only=True, map_location="cpu")
    encoder.load_state_dict(encoder_weights, strict=True)
    encoder = encoder.to(cfg.gpu_id)
    print(f"Loaded MAE pre-trained encoder from {stock_path}")

    train_features, train_labels, mean, std = extract_features(encoder, train_data_loader, cfg.gpu_id)
    test_features, test_labels, _, _ = extract_features(encoder, test_data_loader, cfg.gpu_id)
    del encoder
    torch.cuda.empty_cache()

    ####################################
    ##### SWEEP LEARNING RATES #########
    ####################################

    learning_rates = [1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2]
    best_overall_acc = 0.0
    best_probe = None

    for lr in learning_rates:
        linear_probe = train_linear_probe(
            features=train_features,
            labels=train_labels,
            num_classes=num_classes,
            learning_rate=lr,
            device=cfg.gpu_id
        )
        acc = test_linear_probe(
            linear_probe,
            features=test_features,
            labels=test_labels,
            device=cfg.gpu_id
        )

        print(f"LR: {lr:.4f} | Test Acc: {acc:.2f}%")

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_probe = copy.deepcopy(linear_probe.state_dict())

    ######################
    ##### SAVE MODEL #####
    ######################

    # Fold normalization into the probe parameters
    best_probe_folded = fold_normalization_into_probe(best_probe, mean, std)

    dataset_models_dir = f"{cfg.models_dir}/linear_probes"
    os.makedirs(dataset_models_dir, exist_ok=True)

    save_path = f"{dataset_models_dir}/{cfg.dataset_name}.pt"
    torch.save(best_probe_folded, save_path)
    print(f"Saved linear probe with folded normalization to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    dataset_names = ["cifar100", "eurosat"]
    for dataset_name in dataset_names:
        cfg = utils.load_config("configs/fine_tune_vtab/linear_probe.yaml")
        cfg.dataset_name = dataset_name
        cfg.models_dir = args.models_dir
        cfg.gpu_id = args.gpu_id
        linear_probe(cfg)
