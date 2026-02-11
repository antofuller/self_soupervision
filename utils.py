import torch
import torch.nn.functional as F
import yaml
import os
import csv
from types import SimpleNamespace


@torch.no_grad()
def do_knn_parallel(
    train_embeds,
    train_labels,
    test_embeds,
    test_labels,
    k=20,
    T=0.07,
):
    device = train_embeds.device

    train_labels = train_labels.to(device=device, dtype=torch.long)
    test_labels  = test_labels.to(device=device, dtype=torch.long)

    # infer num_classes from the max label (labels are assumed 0..C-1)
    max_label = torch.maximum(train_labels.max(), test_labels.max())
    num_classes = int(max_label.item()) + 1

    train_embeds = F.normalize(train_embeds, dim=1, p=2)
    test_embeds  = F.normalize(test_embeds,  dim=1, p=2)

    # (N_test, D) @ (D, N_train) -> (N_test, N_train)
    sims = test_embeds @ train_embeds.T

    # top-k neighbors for every test sample: (N_test, k)
    topv, topi = sims.topk(k, dim=1, largest=True, sorted=True)

    # labels of those neighbors: (N_test, k)
    neigh_labels = train_labels[topi]

    # weights: (N_test, k)
    weights = (topv / T).exp()

    # accumulate weighted votes per class
    scores = torch.zeros(
        test_embeds.size(0), num_classes,
        device=device, dtype=weights.dtype
    )
    scores.scatter_add_(dim=1, index=neigh_labels, src=weights)

    preds = scores.argmax(dim=1)
    acc = (preds == test_labels).float().mean().item()
    return acc


def make_warmup_cosine(opt, epochs: int, warmup_frac: float = 0.10):
    warmup_epochs = max(1, int(warmup_frac * epochs))
    cosine_epochs = max(1, epochs - warmup_epochs)

    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_epochs)

    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            elif value == "None":
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, value)

    def as_dict(self):
        return {
            key: value.as_dict() if isinstance(value, NestedNamespace) else value
            for key, value in self.items()
        }

    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        return self.__dict__.items()

    def __setattr__(self, name, value):
        return super().__setattr__(name, value)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return NestedNamespace(config)


def append_to_csv(file_path, input_list):
    # Ensure the parent directory exists (optional but helpful)
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Create the file if it doesn't exist
    if not os.path.exists(file_path):
        open(file_path, "w", newline="").close()

    # Append the row
    with open(file_path, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(input_list)


def test_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100.0
    return acc


def make_weighted_params(state_dicts, weights, gpu_id):
    out = {}
    for key in state_dicts[0].keys():
        weighted_sum = 0
        for i, sd in enumerate(state_dicts):
            param = sd[key].to(gpu_id)
            weighted_sum = weighted_sum + param * weights[i]
        out[key] = weighted_sum
    return out