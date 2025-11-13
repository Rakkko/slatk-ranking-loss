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

from models import MatrixFactorization, LightGCN, XSimGCL
from losses import BPRLoss, SoftmaxLoss
from losses.slatk import ExactSoftmaxLossAtK
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
from math import log


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(num_users: int, num_items: int, **kwargs) -> BaseModel:
    name = kwargs.pop("name").lower()
    if name == "mf":
        return MatrixFactorization(num_users, num_items, kwargs["embedding_dim"], kwargs["user_reg"], kwargs["item_reg"])
    raise ValueError(f"We must use MF for this experiment but found: {name}")


def build_loss(name: str, params: dict) -> nn.Module:
    name = name.lower()
    if name == "bpr":
        return BPRLoss()
    if name == "sl":
        return SoftmaxLoss(tau_d=float(params.get("tau_d", 1.0)))
    if name == "slatk":
        return ExactSoftmaxLossAtK(
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
        "recall": log(rec + 1e-10),
        "ndcg": log(ndcg + 1e-10),
    }


def run_one_step_experiment(cfg: dict):
    """Run experiment with one gradient update step for each loss function."""

    # Set seed for reproducibility
    seed = int(cfg["train"].get("seed"))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
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

    # Build model
    model_kwargs = {"name": "mf"}  # Force the MF model
    model_kwargs.update(cfg["model"])

    # Create initial model with fixed seed
    set_seed(seed)
    model = build_model(num_users, num_items, **model_kwargs).to(device)

    # Save initial state
    initial_state = deepcopy(model.state_dict())

    # Get evaluation K from config
    eval_k = int(cfg["train"]["eval_k"])
    num_negatives = int(cfg["train"]["num_negatives"])
    lr = float(cfg["train"]["lr"])

    print(f"\nExperiment Settings:")
    print(f"- Model: {cfg['model']['name']}")
    print(f"- Evaluation K: {eval_k}")
    print(f"- Learning rate: {lr}")
    print(f"- Number of negatives: {num_negatives}")
    print(f"- Dataset: {cfg['dataset']['type']}")
    print("-" * 50)

    # Results storage
    results = {}

    # Test cases: 3 loss functions
    test_cases = [
        ("no_update", None),
        ("bpr", "bpr"),
        ("softmax", "sl"),
        ("softmax_at_k", "slatk")
    ]

    for case_name, loss_name in test_cases:
        print(f"\nTesting: {case_name}")

        # Reset model to initial state
        model.load_state_dict(deepcopy(initial_state))

        if loss_name is not None:
            # Build loss function
            loss_params = cfg["train"].get("loss_params", {})
            if loss_name == "slatk":
                loss_params["topk"] = eval_k  # Use eval_k as topk for SLatK
            loss_fn = build_loss(loss_name, loss_params)

            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Get one batch and perform one gradient step
            model.train()
            for batch in train_loader:
                user_ids, pos_ids = [x.to(device) for x in batch]

                # Negative sampling
                neg_ids = neg_sampler.sample(user_ids, num_negatives).to(device)

                # Compute scores
                pos_scores = model(user_ids, pos_ids)
                neg_scores = model(
                    user_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1),
                    neg_ids.reshape(-1)
                ).reshape(-1, num_negatives)

                all_scores = model.full_item_scores(user_ids)

                # Compute loss
                loss = loss_fn(
                    user_ids,
                    pos_ids,
                    pos_scores=pos_scores,
                    neg_item_ids=neg_ids,
                    neg_scores=neg_scores,
                    all_item_scores=all_scores
                )

                # Add L2 regularization
                total_loss = loss

                # One gradient step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                print(f"  Loss value: {float(total_loss.detach().cpu()):.6f}")
                break  # Only one step

        # Evaluate
        val_metrics = evaluate(model, val_dict, eval_k, device)
        test_metrics = evaluate(model, test_dict, eval_k, device)

        results[case_name] = {
            "val": val_metrics,
            "test": test_metrics
        }

        print(f"  Validation - Recall@{eval_k}: {val_metrics['recall']:.6f}, NDCG@{eval_k}: {val_metrics['ndcg']:.6f}")
        print(f"  Test - Recall@{eval_k}: {test_metrics['recall']:.6f}, NDCG@{eval_k}: {test_metrics['ndcg']:.6f}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY - One Gradient Step Effect (K={eval_k})")
    print("=" * 70)
    print(f"{'Method':<15} {'Val Recall@K':<15} {'Val NDCG@K':<15} {'Test Recall@K':<15} {'Test NDCG@K':<15}")
    print("-" * 70)

    for case_name, metrics in results.items():
        print(f"{case_name:<15} "
              f"{metrics['val']['recall']:<15.6f} "
              f"{metrics['val']['ndcg']:<15.6f} "
              f"{metrics['test']['recall']:<15.6f} "
              f"{metrics['test']['ndcg']:<15.6f}")

    return results


def main():
    cfg = load_config("cfgs/figure1.yaml")
    results = run_one_step_experiment(cfg)


if __name__ == "__main__":
    main()
