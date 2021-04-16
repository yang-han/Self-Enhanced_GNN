import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models import GAT, GCN, SGC, USGC
from utils import (EPOCHS_CONFIG, EdgesAdder, EdgesDropper2, EdgesModifier,
                   load_dataset, load_split, test, train)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--model", type=str, default="GCN")
    # parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--splits-dir", type=str, default="splits")
    parser.add_argument(
        "--mode", type=str, choices=["adder", "dropper2", "modifier"], required=True
    )
    parser.add_argument("--pre-seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--normalization", action="store_true")
    parser.add_argument("--out-dir", type=str, required=True)

    return parser


def main(_args=None):
    args = create_parser().parse_args(_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.dataset)
    data = dataset[0]

    model_name = args.model.upper()
    if model_name == "SGC" and args.dataset.lower() in ["cs", "physics"]:
        model_name = "USGC"
        print(model_name, end=" ")
    model_cls = globals()[model_name]
    data = data.to(device)
    edges = data.edge_index.to(device)

    train_mask, val_mask, test_mask = load_split(
        f"{args.splits_dir}/{args.dataset.lower()}_{args.split_id}.mask"
    )
    epochs = EPOCHS_CONFIG[args.dataset.lower()]

    print("Preparing Model...")
    torch.manual_seed(args.pre_seed)
    model = model_cls(dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in range(epochs):
        train(model, optimizer, data, edges, train_mask)
    print("Done! ", test(model, data, edges, train_mask, val_mask, test_mask))
    sys.stdout.flush()

    y = F.softmax(model(data.x, edges), dim=1)

    modifier_cls = globals()["Edges" + args.mode.capitalize()]
    modifier = modifier_cls(y, edges, device, normalization=args.normalization)

    all_thresholds = [0.01 * x for x in range(101)]

    dataset_dir = os.path.join(
        args.out_dir, args.dataset.lower(), args.mode.lower())
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    adj_m = None
    for threshold in all_thresholds:
        print(f"{args.dataset} {args.model} {args.split_id} {threshold:.2f}", end=" ")
        save_path = os.path.join(
            dataset_dir, f"{args.model.lower()}_{args.split_id}_{threshold:.2f}.edges"
        )
        adj_m = modifier.modify(threshold, adj_m)
        new_edges = adj_m.nonzero().t().to(device)
        torch.save(new_edges, save_path)
        print("|", datetime.now(), new_edges.size(1))


if __name__ == "__main__":
    main()
