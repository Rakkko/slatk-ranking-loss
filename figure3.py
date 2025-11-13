from __future__ import annotations

import argparse
from copy import deepcopy
import random
from typing import Dict, Set

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MatrixFactorization
from losses import SoftmaxLossAtK
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


def evaluate_at_k(model: BaseModel, user_to_eval_pos: Dict[int, Set[int]],
                  k_values: list[int], device: torch.device) -> dict[int, dict[str, float]]:
    """Evaluate model at multiple K values."""
    model.eval()
    results = {}

    with torch.no_grad():
        users = sorted(user_to_eval_pos.keys())
        if not users:
            return results

        user_ids = torch.tensor(users, dtype=torch.long, device=device)
        scores = model.full_item_scores(user_ids)
        gt = [list(user_to_eval_pos[u]) for u in users]

        for k in k_values:
            results[k] = {
                "recall": recall_at_k(scores, gt, k),
                "ndcg": ndcg_at_k(scores, gt, k)
            }

    return results


def train_slatk_with_k(
    train_k: int,
    cfg: dict,
    train_loader: DataLoader,
    val_dict: Dict[int, Set[int]],
    user_pos_dict: Dict[int, Set[int]],
    neg_sampler,
    num_users: int,
    num_items: int,
    device: torch.device,
    seed: int,
    epochs: int = 10
) -> dict:
    """Train model with SL@K loss at specific K value."""

    # Set seed for reproducible initialization
    set_seed(seed)

    # Build model
    embedding_dim = 64
    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        user_reg=cfg["model"].get("user_reg", 0.0),
        item_reg=cfg["model"].get("item_reg", 0.0)
    ).to(device)

    # Build SL@K loss with specific K
    loss_params = cfg["train"].get("loss_params", {})
    loss_fn = SoftmaxLossAtK(
        topk=train_k,
        tau_d=float(loss_params.get("tau_d", 1.0)),
        tau_w=float(loss_params.get("tau_w", 1.0))
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    num_negatives = int(cfg["train"]["num_negatives"])

    # Train
    print(f"\nTraining with SL@{train_k}:")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False)
        for batch in progress_bar:
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

            # Add L2 regularization
            reg = model.l2_regularization(user_ids, pos_ids, neg_ids)
            total_loss = loss + reg

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.detach().cpu())

        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch}: Loss={avg_loss:.4f}")

    # Evaluate at multiple K values
    eval_k_values = [5, 10, 20]
    final_metrics = evaluate_at_k(model, val_dict, eval_k_values, device)

    return final_metrics


def run_slatk_matching_experiment(cfg: dict):
    """Test SL@K with different training K values against different evaluation K values."""

    # Set seed
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
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
    print("SL@K MATCHING EXPERIMENT")
    print("Testing if SL@K performs best when training K matches evaluation K")
    print("Dataset: MovieLens 100K, Model: MF (64D), Epochs: 10")
    print("=" * 70)

    # Training and evaluation K values
    train_k_values = [5, 10, 20]
    eval_k_values = [5, 10, 20]

    # Store results
    results = {}

    # Train models with different K values
    for train_k in train_k_values:
        metrics = train_slatk_with_k(
            train_k=train_k,
            cfg=cfg,
            train_loader=train_loader,
            val_dict=val_dict,
            user_pos_dict=user_pos_dict,
            neg_sampler=neg_sampler,
            num_users=num_users,
            num_items=num_items,
            device=device,
            seed=seed,
            epochs=10
        )
        results[train_k] = metrics

    # Print results in table format
    print("\n" + "=" * 70)
    print("RESULTS: NDCG@K Performance")
    print("=" * 70)

    # Print header
    print(f"{'':>10}", end="")
    for eval_k in eval_k_values:
        print(f"{'NDCG@' + str(eval_k):>15}", end="")
    print()

    print("-" * 60)

    # Print rows
    for train_k in train_k_values:
        print(f"{'SL@' + str(train_k):>10}", end="")
        for eval_k in eval_k_values:
            ndcg_value = results[train_k][eval_k]["ndcg"]
            print(f"{ndcg_value:>15.6f}", end="")
        print()

    # Highlight diagonal performance
    print("\n" + "=" * 70)
    print("DIAGONAL PERFORMANCE (Training K = Evaluation K)")
    print("=" * 70)

    for k in train_k_values:
        ndcg_value = results[k][k]["ndcg"]
        print(f"SL@{k} evaluated at NDCG@{k}: {ndcg_value:.6f}")

    # Calculate and show relative improvements
    print("\n" + "=" * 70)
    print("RELATIVE PERFORMANCE (% of best in column)")
    print("=" * 70)

    print(f"{'':>10}", end="")
    for eval_k in eval_k_values:
        print(f"{'NDCG@' + str(eval_k):>15}", end="")
    print()

    print("-" * 60)

    for train_k in train_k_values:
        print(f"{'SL@' + str(train_k):>10}", end="")
        for eval_k in eval_k_values:
            # Find best performance in this column
            best_ndcg = max(results[tk][eval_k]["ndcg"] for tk in train_k_values)
            current_ndcg = results[train_k][eval_k]["ndcg"]
            relative_perf = (current_ndcg / best_ndcg) * 100 if best_ndcg > 0 else 0

            # Highlight if this is the best
            if current_ndcg == best_ndcg:
                print(f"{'*' + f'{relative_perf:.1f}%':>14}", end="")
            else:
                print(f"{relative_perf:>14.1f}%", end="")
        print()

    print("\n* indicates best performance in column")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfgs/figure3.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override model to ensure we use MF
    cfg["model"]["name"] = "mf"
    cfg["model"]["embedding_dim"] = 64

    results = run_slatk_matching_experiment(cfg)


if __name__ == "__main__":
    main()
