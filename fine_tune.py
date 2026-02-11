import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import copy

import data
import utils
import models


def fine_tune(cfg):
    ########################
    ##### PREPARE DATA #####
    ########################

    train_dataset = data.miniVTAB(
        dataset_name=cfg.dataset_name,
        split="train",
        augmentation=cfg.augmentation,
        do_two_views=False
    )
    num_classes = train_dataset.get_num_classes()
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Dataset length: {len(train_dataset)}")
    print(f"Number of classes: {num_classes}")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
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

    #######################################
    ##### PREPARE MODEL and OPTIMIZER #####
    #######################################

    encoder_params = torch.load(cfg.encoder_path, weights_only=True, map_location="cpu")
    print(f"Loaded encoder from {cfg.encoder_path}")

    head_path = f"{cfg.models_dir}/linear_probes/{cfg.dataset_name}.pt"
    head_params = torch.load(head_path, weights_only=True, map_location="cpu")

    model = models.ViTClassifier(encoder_params, head_params, num_classes).to(cfg.gpu_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    steps_per_epoch = len(train_data_loader)
    total_steps = cfg.epochs * steps_per_epoch
    schedule = utils.make_warmup_cosine(optimizer, epochs=total_steps, warmup_frac=0.10)
    criterion = nn.CrossEntropyLoss()

    #################
    ##### TRAIN #####
    #################

    model.train()
    best_acc = 0.0
    best_model = None

    for epoch in range(cfg.epochs):
        for batch in train_data_loader:
            optimizer.zero_grad()

            images, labels = batch
            images = images.to(cfg.gpu_id)
            labels = labels.to(cfg.gpu_id)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            schedule.step()

        # Test every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = utils.test_model(model, test_data_loader, cfg.gpu_id)
            print(f"Epoch {epoch+1}/{cfg.epochs} | Test Acc: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model.state_dict())

            model.train()

    # Final test
    acc = utils.test_model(model, test_data_loader, cfg.gpu_id)
    print(f"Final Test Acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_model = copy.deepcopy(model.state_dict())

    ######################
    ##### SAVE MODEL #####
    ######################

    ingredient_name = os.path.basename(cfg.encoder_path).replace(".pth", "")
    fine_tuned_dir = f"{cfg.models_dir}/fine_tuned_vtab/{cfg.dataset_name}/{ingredient_name}"
    os.makedirs(fine_tuned_dir, exist_ok=True)

    ft_name = f"ft_lr={cfg.learning_rate}_epochs={cfg.epochs}"
    save_path = f"{fine_tuned_dir}/{ft_name}.pt"
    torch.save(best_model, save_path)
    print(f"Best model saved to {save_path} with accuracy {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    exp_name = "inter_train_vtab_for_ft"
    ingredient_cfg_dir = f"configs/{exp_name}"
    ingredient_cfg_names = os.listdir(ingredient_cfg_dir)
    ft_cfg_names = os.listdir("configs/fine_tune_vtab")

    dataset_names = [
        "cifar100",
        "eurosat"
    ]
    for dataset_name in dataset_names:
        for ingredient_cfg_name in ingredient_cfg_names:
            model_cfg = utils.load_config(os.path.join(ingredient_cfg_dir, ingredient_cfg_name))
            model_cfg.steps = 100  # override for testing
            model_cfg_str = "_".join(f"{k}={v}" for k, v in model_cfg.model_cfg.items())
            path = (
                f"{args.models_dir}/{exp_name}/{dataset_name}_intertrainings/"
                f"{model_cfg_str}"
                f"_lrb={model_cfg.learning_rate_backbone}"
                f"_lrh={model_cfg.learning_rate_head}"
                f"_steps={model_cfg.steps}"
                f"_bs={model_cfg.batch_size}"
                f"_aug={model_cfg.augmentation}"
                f".pt"
            )
            
            for ft_cfg_name in ft_cfg_names:
                ft_cfg = utils.load_config(os.path.join("configs/fine_tune_vtab", ft_cfg_name))
                ft_cfg.dataset_name = dataset_name
                ft_cfg.models_dir = args.models_dir
                ft_cfg.encoder_path = path
                ft_cfg.gpu_id = args.gpu_id

                fine_tune(ft_cfg)

