import os
import torch
from torch.utils.data import DataLoader
from torch.func import functional_call
import torch.nn.functional as F
import copy
import argparse

import data
import utils
import models


def knn_inbatch_neighbor_entropy(batch_embeds, k=20, T=0.07):
    B = batch_embeds.shape[0]

    z = F.normalize(batch_embeds.float(), dim=1, p=2)
    sim = z @ z.T
    sim.fill_diagonal_(float("-inf"))

    k_eff = min(k, B - 1)
    topv, _ = sim.topk(k_eff, dim=1, largest=True, sorted=False)

    logits = topv / T
    p = logits.softmax(dim=1)
    entropy = -(p * (p.clamp_min(1e-12)).log()).sum(1)
    return entropy.mean()


class MixtureCoefficients(torch.nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n, device=device))

    def forward(self):
        return self.logits.softmax(dim=0)


def forward_with_soup(model, images, weighted_params):
    buffers = dict(model.named_buffers())
    return functional_call(model, (weighted_params, buffers), (images,))


def make_weighted_params(state_dicts, weights, gpu_id):
    out = {}
    for key in state_dicts[0].keys():
        weighted_sum = 0
        for i, sd in enumerate(state_dicts):
            param = sd[key].to(gpu_id)
            weighted_sum = weighted_sum + param * weights[i]
        out[key] = weighted_sum
    return out


def load_checkpoints(model_paths, gpu_id):
    all_checkpoints = []
    for path in model_paths:
        try:
            weights = torch.load(path, weights_only=True, map_location="cpu")
            weights = {k: v.to(gpu_id) for k, v in weights.items()}
            all_checkpoints.append(weights)
            print(f"Loaded: {path}")
        except:
            print(f"FAILED: {path}")
    return all_checkpoints


def get_embeds(model, loader, gpu_id):
    model.eval()
    all_embeds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(gpu_id)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embeds = model(images)[:, 0, :]
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)


def test_soup(model, train_loader, test_loader, gpu_id):
    train_embeds, train_labels = get_embeds(model, train_loader, gpu_id)
    test_embeds, test_labels = get_embeds(model, test_loader, gpu_id)
    return utils.do_knn_parallel(
        train_embeds.to(gpu_id), train_labels.to(gpu_id),
        test_embeds.to(gpu_id), test_labels.to(gpu_id)
    )


def self_season(cfg):
    ########################
    ##### PREPARE DATA #####
    ########################

    torch.multiprocessing.set_sharing_strategy('file_system')
    
    train_dataset = data.miniVTAB(
        dataset_name=cfg.dataset_name,
        split="train",
        augmentation="none",
        do_two_views=False
    )
    train_loader = DataLoader(
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    #######################################
    ##### PREPARE MODEL and OPTIMIZER #####
    #######################################

    all_checkpoints = load_checkpoints(cfg.model_paths, cfg.gpu_id)
    model = models.vit_base().to(cfg.gpu_id)
    
    coefficients = MixtureCoefficients(len(all_checkpoints), device=cfg.gpu_id)
    optimizer = torch.optim.Adam([coefficients.logits], lr=cfg.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    #################
    ##### TRAIN #####
    #################

    best_loss = float('inf')
    best_coefficients = None
    model.eval()

    for epoch in range(cfg.epochs):
        losses = []
        for images, _ in train_loader:
            optimizer.zero_grad()
            images = images.to(cfg.gpu_id)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                wparams = make_weighted_params(all_checkpoints, coefficients(), cfg.gpu_id)
                embeds = forward_with_soup(model, images, wparams)[:, 0, :]

            loss = knn_inbatch_neighbor_entropy(embeds, k=cfg.k, T=cfg.temp)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_coefficients = copy.deepcopy(coefficients().detach())

        schedule.step()
        mix = [f"{x:.3f}" for x in coefficients().detach().cpu().tolist()]
        print(f"Epoch {epoch}/{cfg.epochs} | Loss: {avg_loss:.4f} | Mix: {mix}")

    ################
    ##### TEST #####
    ################

    best_mixed_weights = make_weighted_params(all_checkpoints, best_coefficients, cfg.gpu_id)
    model.load_state_dict(best_mixed_weights)
    test_acc = test_soup(model, train_loader, test_loader, cfg.gpu_id)
    print(f"Test accuracy: {test_acc:.4f}")

    ################
    ##### SAVE #####
    ################

    result_file = f"{cfg.save_dir}/self_seasoning_results.csv"
    utils.append_to_csv(result_file, [cfg.dataset_name, cfg.k, cfg.lr, cfg.epochs, best_coefficients.tolist(), test_acc])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing saved models")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    ingredient_cfg_names = [
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
        cfg = utils.load_config("configs/self_season/default.yaml")
        cfg.dataset_name = dataset_name
        cfg.models_dir = args.models_dir
        cfg.save_dir = args.save_dir
        
        # Build model paths from inter-training outputs
        cfg.model_paths = [f"{cfg.models_dir}/mae_encoder.pth"]  # stock
        for ingredient_cfg_name in ingredient_cfg_names:
            model_cfg = utils.load_config(f"configs/inter_train_vtab_season/{ingredient_cfg_name}.yaml")
            model_cfg_str = "_".join(f"{k}={v}" for k, v in model_cfg.model_cfg.items())
            path = (
                f"{cfg.models_dir}/{dataset_name}_intertrainings/"
                f"{model_cfg_str}"
                f"_lrb={model_cfg.learning_rate_backbone}"
                f"_lrh={model_cfg.learning_rate_head}"
                f"_steps={model_cfg.steps}"
                f"_bs={model_cfg.batch_size}"
                f"_aug={model_cfg.augmentation}"
                f".pt"
            )
            cfg.model_paths.append(path)
        
        self_season(cfg)