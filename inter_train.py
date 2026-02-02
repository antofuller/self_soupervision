import os
import torch
import yaml
from torch.utils.data import DataLoader
import argparse

import data
import utils
import models

def inter_train(cfg):
    ########################
    ##### PREPARE DATA #####
    ########################

    do_two_views = cfg.model_cfg.algo != "mae"
    dataset = data.miniVTAB(
        dataset_name=cfg.dataset_name,
        split="train",
        augmentation=cfg.augmentation,
        do_two_views=do_two_views
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    #######################################
    ##### PREPARE MODEL and OPTIMIZER #####
    #######################################

    model = models.fetch_model_for_inter_training(cfg.model_cfg, cfg.models_dir).to(cfg.gpu_id)

    head_params = list(model.head.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    num_backbone_params = sum(p.numel() for p in backbone_params)
    num_head_params = sum(p.numel() for p in head_params)
    print(f"Backbone params: {num_backbone_params/1e6:.2f}M | Head params: {num_head_params/1e6:.2f}M")
    
    backbone_optimizer = torch.optim.AdamW(backbone_params, lr=cfg.learning_rate_backbone, weight_decay=0.01)
    head_optimizer = torch.optim.AdamW(head_params, lr=cfg.learning_rate_head, weight_decay=0.01)

    backbone_schedule = utils.make_warmup_cosine(backbone_optimizer, epochs=cfg.steps, warmup_frac=0.10)  # 10% because why not?
    head_schedule = utils.make_warmup_cosine(head_optimizer, epochs=cfg.steps, warmup_frac=0.01)  # 1% because why not?

    #################
    ##### TRAIN #####
    #################

    model.train()
    total_loss = 0.0
    step = 0
    print_every = max(1, int(0.01 * cfg.steps))

    while step < cfg.steps:
        for batch in data_loader:
            backbone_optimizer.zero_grad()
            head_optimizer.zero_grad()
            
            if len(batch) == 3:  # for two-view algorithms
                inputs = (batch[0].to(cfg.gpu_id), batch[1].to(cfg.gpu_id))
            else:  # for single-view algorithms
                inputs = (batch[0].to(cfg.gpu_id),)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(*inputs)
            
            loss.backward()

            backbone_optimizer.step()
            head_optimizer.step()

            total_loss += loss.item()
            backbone_schedule.step()
            head_schedule.step()
            step += 1

            if step % print_every == 0:
                avg_loss = total_loss / print_every
                print(f"Step {step}/{cfg.steps} | Loss: {avg_loss:.4f} | LR backbone: {backbone_schedule.get_last_lr()[0]:.6f} | LR head: {head_schedule.get_last_lr()[0]:.6f}")
                total_loss = 0.0
                
            if step >= cfg.steps:
                break

    ######################
    ##### SAVE MODEL #####
    ######################

    dataset_models_dir = f"{cfg.models_dir}/{cfg.dataset_name}_intertrainings"
    os.makedirs(dataset_models_dir, exist_ok=True)
    model_cfg_str = "_".join(f"{k}={v}" for k, v in cfg.model_cfg.items())

    save_path = (
        f"{dataset_models_dir}/"
        f"{model_cfg_str}"
        f"_lrb={cfg.learning_rate_backbone}"
        f"_lrh={cfg.learning_rate_head}"
        f"_steps={cfg.steps}"
        f"_bs={cfg.batch_size}"
        f"_aug={cfg.augmentation}"
        f".pt"
    )

    torch.save(model.encoder.state_dict(), save_path)
    print(f"Backbone saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True, help="Directory to save models")
    args = parser.parse_args()

    cfg_names = [
        "mae",
        "mmcr_global_local",
        "mmcr_global_only",
        "mocov3_T=0.1",
        "mocov3_T=1.0"
    ]
    dataset_names = [
        "cifar100",
        "eurosat"
    ]
    for dataset_name in dataset_names:
        for cfg_name in cfg_names:
            cfg = utils.load_config(f"configs/inter_train_vtab_season/{cfg_name}.yaml")
            cfg.dataset_name = dataset_name
            cfg.models_dir = args.models_dir
            inter_train(cfg)