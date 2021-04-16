import argparse
import os
import pickle

import torch

from utils import ALL_DATASETS, Mask, generate_split, load_dataset


def create_parser():
    parser = argparse.ArgumentParser(description="Generate random splits.")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="splits")

    return parser


def save(dataset, seed, path):
    mask = Mask(*generate_split(dataset, seed))
    torch.save(mask, path)


def split():
    args = create_parser().parse_args()
    seeds = list(range(args.num_seeds))
    out_dir = args.out_dir
    if args.dataset == "all":
        for dataset in ALL_DATASETS:
            print(f"{dataset}...", end=" ")
            dataset_set = load_dataset(dataset)
            data = dataset_set[0]
            num_nodes = data.y.size(0)
            num_classes = dataset_set.num_classes
            num_train = (num_nodes // num_classes) // 10
            print("num_train: ", num_train, end=" ")
            for seed in seeds:
                mask = Mask(*generate_split(dataset, seed, num_train, num_train))
                torch.save(mask, f"{out_dir}/{dataset}_{seed}.mask")
            print("Done")
    else:
        dataset = args.dataset
        for seed in seeds:
            mask = Mask(*generate_split(dataset, seed, num_train, num_train))
            torch.save(mask, f"{out_dir}/{dataset}_{seed}.mask")


if __name__ == "__main__":
    split()
