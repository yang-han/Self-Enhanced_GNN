import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models import GAT, GCN, SGC, USGC
from utils import (EPOCHS_CONFIG, load_data, load_dataset, load_split, test,
                   train)


def create_parser():
    parser = argparse.ArgumentParser(description="train many times.")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--model", type=str, default="GCN")
    # parser.add_argument("--edge-prefix", type=str)
    # parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--edges-dir", type=str, default="6_edges")
    parser.add_argument("--splits-dir", type=str, default="splits")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["adder", "dropper", "dropper2", "modifier"],
        required=True,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--out-dir", type=str, default="exp_6")

    parser.add_argument("--resume", type=int)
    return parser


def main(_args=None):
    args = create_parser().parse_args(_args)

    epochs = EPOCHS_CONFIG[args.dataset.lower()]
    num_seeds = args.num_seeds
    num_splits = args.num_splits

    model_name = args.model.upper()
    if model_name == "SGC" and args.dataset.lower() in ["cs", "physics"]:
        model_name = "USGC"
        print(model_name, end=" ")
    if (
        model_name == "SGC"
        and args.mode == "adder"
        and args.dataset.lower() in ["computers"]
    ):
        model_name = "USGC"
        print(model_name, end=" ")
    model_cls = globals()[model_name]

    num_thresholds = 101
    if model_name == "GAT" and args.dataset == "physics":
        num_thresholds = 60

    seeds = list(range(num_seeds))
    splits = list(range(num_splits))

    dataset = load_dataset(args.dataset)
    data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edges = data.edge_index.to(device)

    if args.output is None:
        output = f"{args.out_dir}/{args.model.lower()}_{args.dataset.lower()}_{args.mode}_{num_splits}_{num_seeds}.np"
    elif args.output == "time":
        output = "{time}.np".format(
            time=datetime.now().strftime(r"%Y-%m-%d_%H:%M:%S"))
    else:
        output = args.output

    result = np.zeros((3, 101, num_splits, num_seeds, epochs)
                      )  # 101: num_of_thresholds

    if args.resume is not None:
        result = load_data(output)
        splits = list(range(args.resume, num_splits))

    start_time = datetime.now()
    for split in splits:
        train_mask, val_mask, test_mask = load_split(
            f"{args.splits_dir}/{args.dataset.lower()}_{split}.mask"
        )
        for threshold_i in range(num_thresholds):
            threshold = 0.01 * threshold_i
            for seed in seeds:
                torch.manual_seed(seed)  # set seed here

                edges = torch.load(
                    os.path.join(
                        args.edges_dir,
                        args.dataset.lower(),
                        args.mode,
                        f"{args.model.lower()}_{split}_{threshold:.2f}.edges",
                    )
                )
                model = model_cls(dataset.num_features,
                                  dataset.num_classes).to(device)
                data = data.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.01, weight_decay=5e-4
                )

                best_val_acc = test_acc = 0
                for epoch in range(epochs):
                    train(model, optimizer, data, edges, train_mask)
                    train_acc, val_acc, tmp_test_acc = test(
                        model, data, edges, train_mask, val_mask, test_mask
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        test_acc = tmp_test_acc
                    log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"

                    result[0][threshold_i][split][seed][epoch] = train_acc
                    result[1][threshold_i][split][seed][epoch] = best_val_acc
                    result[2][threshold_i][split][seed][epoch] = test_acc

                    if args.verbose:
                        print(
                            f"{datetime.now()}",
                            log.format(epoch + 1, train_acc,
                                       best_val_acc, test_acc),
                        )

                    if epoch == epochs - 1:
                        now = datetime.now()
                        print(
                            f"{now} {args.model.lower()} split: {split} threshold: {threshold:.2f} seed: {seed}",
                            log.format(epoch + 1, train_acc,
                                       best_val_acc, test_acc),
                            f"Elapsed: {now - start_time}",
                        )
                sys.stdout.flush()
        # per-split
        with open(output, "wb") as f:
            pickle.dump(result, f)

    print(result)

    # input("Waiting for writing into disk...\nPlease press <ENTER>")
    # with open(output, "wb") as f:
    #     pickle.dump(result, f)


if __name__ == "__main__":
    main()
