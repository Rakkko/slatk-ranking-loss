from __future__ import annotations

import argparse
from copy import deepcopy
import os
import random
from typing import Dict, Set

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MatrixFactorization
from losses import BPRLoss, SoftmaxLoss, SoftmaxLossAtK
from metrics import recall_at_k, ndcg_at_k
from data.movielens import (
    TripletDataset,
    load_ml100k_interactions,
    split_leave_one_out,
    build_user_pos_dict,
)
from data.read_proc_data import get_data_summary, load_proc_data
from samplers import UniformNegativeSampler
from models.base import BaseModel


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_loss(name: str, params: dict) -> nn.Module:
    name = name.lower()
    if name == "bpr":
        return BPRLoss()
    if name == "sl":
        return SoftmaxLoss(tau_d=float(params.get("tau_d", 1.0)))
    if name == "slatk":
        return SoftmaxLossAtK(
            topk=int(params.get("topk", 10)),
            tau_d=float(params.get("tau_d", 1.0)),
            tau_w=float(params.get("tau_w", 1.0)),
        )
    raise ValueError(f"Unknown loss: {name}")


def evaluate(model: BaseModel, user_to_eval_pos: Dict[int, Set[int]], k: int, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        users = sorted(user_to_eval_pos.keys())
        if not users:
            return {}
        user_ids = torch.tensor(users, dtype=torch.long, device=device)
        scores = model.full_item_scores(user_ids)
        gt = [list(user_to_eval_pos[u]) for u in users]
        rec = recall_at_k(scores, gt, k)
        ndcg = ndcg_at_k(scores, gt, k)
    return {
        "recall": rec,
        "ndcg": ndcg,
    }


def train_model_with_loss(
    cfg: dict,
    loss_name: str,
    train_loader: DataLoader,
    val_dict: Dict[int, Set[int]],
    user_pos_dict: Dict[int, Set[int]],
    neg_sampler,
    num_users: int,
    num_items: int,
    device: torch.device,
    seed: int
) -> dict:
    """Train MF model with specified loss for 10 epochs."""

    # Fixed parameters for the experiment
    embedding_dim = cfg["model"].get("embedding_dim", 64)
    epochs = cfg["train"].get("epochs", 10)
    eval_k = cfg["train"].get("eval_k", 10)

    # Set seed for reproducible initialization
    set_seed(seed)

    # Build MF model
    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        user_reg=cfg["model"].get("user_reg", 0.0),
        item_reg=cfg["model"].get("item_reg", 0.0)
    ).to(device)

    # Build loss function
    loss_params = cfg["train"].get("loss_params", {})
    if loss_name == "slatk":
        loss_params["topk"] = eval_k  # Use eval_k as topk for SLatK
    loss_fn = build_loss(loss_name, loss_params)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    num_negatives = int(cfg["train"]["num_negatives"])

    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "val_recall": [],
        "val_ndcg": []
    }

    print(f"\nTraining with {loss_name.upper()} loss:")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            user_ids, pos_ids = [x.to(device) for x in batch]

            # Negative sampling
            neg_ids = neg_sampler.sample(user_ids, num_negatives).to(device)

            # Compute scores
            pos_scores = model(user_ids, pos_ids)
            neg_scores = model(
                user_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1),
                neg_ids.reshape(-1)
            ).reshape(-1, num_negatives)

            # Compute loss
            loss = loss_fn(
                user_ids,
                pos_ids,
                pos_scores=pos_scores,
                neg_item_ids=neg_ids,
                neg_scores=neg_scores,
            )

            # reg = model.l2_regularization(user_ids, pos_ids, neg_ids)
            total_loss = loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.detach().cpu())

        # Evaluate on validation set
        val_metrics = evaluate(model, val_dict, eval_k, device)

        avg_loss = epoch_loss / len(train_loader)
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_recall"].append(val_metrics["recall"])
        history["val_ndcg"].append(val_metrics["ndcg"])

        print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Val Recall@{eval_k}={val_metrics['recall']:.4f}, Val NDCG@{eval_k}={val_metrics['ndcg']:.4f}")

    return history


def run_loss_comparison_experiment(cfg: dict):
    """Compare different loss functions on MovieLens 100K with MF."""

    # Set seed
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MovieLens 100K data
    if cfg["dataset"]["type"] == "proc":
        num_users, num_items = get_data_summary(cfg["dataset"]["root"])
        train_pairs, val_dict, test_dict = load_proc_data(cfg["dataset"]["root"])
    else:
        interactions, num_users, num_items = load_ml100k_interactions(
            cfg["dataset"]["root"],
            cfg["dataset"]["threshold"]
        )
        train_pairs, val_dict, test_dict = split_leave_one_out(interactions, num_users)

    train_ds = TripletDataset(train_pairs)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        drop_last=False,
    )

    user_pos_dict = build_user_pos_dict(train_pairs, num_users)
    neg_sampler = UniformNegativeSampler(num_items, user_pos_dict)

    print("=" * 70)
    print("LOSS FUNCTION COMPARISON EXPERIMENT")
    print("Dataset: MovieLens 100K")
    print("Model: Matrix Factorization (64D)")
    print("Training: 10 epochs")
    print("Evaluation: Recall@K and NDCG@K on validation set")
    print("=" * 70)

    # Loss functions to compare
    loss_functions = ["slatk", "bpr", "sl"]
    results = {}

    for loss_name in loss_functions:
        history = train_model_with_loss(
            cfg=cfg,
            loss_name=loss_name,
            train_loader=train_loader,
            val_dict=val_dict,
            user_pos_dict=user_pos_dict,
            neg_sampler=neg_sampler,
            num_users=num_users,
            num_items=num_items,
            device=device,
            seed=seed
        )
        results[loss_name] = history

    # Print final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Validation Set)")
    print("=" * 70)
    print(f"{'Loss Function':<15} {'Recall@K':<12} {'NDCG@K':<12}")
    print("-" * 40)

    for loss_name in loss_functions:
        final_recall = results[loss_name]["val_recall"][-1]
        final_ndcg = results[loss_name]["val_ndcg"][-1]
        print(f"{loss_name.upper():<15} {final_recall:<12.6f} {final_ndcg:<12.6f}")

    # Print epoch-by-epoch comparison
    print("\n" + "=" * 70)
    print("EPOCH-BY-EPOCH COMPARISON")
    print("=" * 70)

    for epoch in range(1, 11):
        print(f"\nEpoch {epoch}:")
        print(f"{'Loss':<10} {'Recall@K':<12} {'NDCG@K':<12}")
        print("-" * 35)
        for loss_name in loss_functions:
            recall = results[loss_name]["val_recall"][epoch-1]
            ndcg = results[loss_name]["val_ndcg"][epoch-1]
            print(f"{loss_name.upper():<10} {recall:<12.6f} {ndcg:<12.6f}")

    return results


def main():
    cfg = load_config("cfgs/figure2.yaml")

    # Override model to ensure we use MF
    cfg["model"]["name"] = "mf"
    cfg["model"]["embedding_dim"] = 64

    results = run_loss_comparison_experiment(cfg)


if __name__ == "__main__":
    main()
