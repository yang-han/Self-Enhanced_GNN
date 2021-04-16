import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models import GAT, GCN, SGC, USGC
from utils import (EPOCHS_CONFIG, load_data, load_dataset, load_split, test,
                   train)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--model", type=str, default="GCN")
    # parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-seeds", type=int, default=10)
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--splits-dir", type=str, default="splits")
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--num-models", type=int, default=2)
    parser.add_argument("--template", type=str)
    parser.add_argument("--filename", type=str)

    # parser.add_argument("--thres-i", type=int, default=90)
    # parser.add_argument("--thres-j", type=int, default=90)
    # parser.add_argument("--enlarge", type=float, default=0.5)
    parser.add_argument("--verbose", "-v", action="store_true")
    # parser.add_argument("--output", type=str)

    return parser


def _main(_args=None):
    args = create_parser().parse_args(_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = EPOCHS_CONFIG[args.dataset.lower()]

    model_name = args.model.upper()
    if model_name == "SGC" and args.dataset.lower() in ["cs", "physics"]:
        model_name = "USGC"
        print(model_name, end=" ")

    dataset = load_dataset(args.dataset.lower())
    data = dataset[0]
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes
    # num_thres = args.num_thres
    edges = data.edge_index.to(device)
    data = data.to(device)
    model_cls = globals()[model_name]
    # print("num_nodes", data.num_nodes)
    o_train_mask, o_val_mask, o_test_mask = load_split(
        f"{args.splits_dir}/{args.dataset.lower()}_{args.split_id}.mask"
    )

    # o_seen_mask = o_train_mask + o_val_mask
    # seen_index = o_seen_mask.nonzero()

    o_y = data.y.clone()
    # seeds = list(2 ** s - 1 for s in range(args.num_models, 0, -1))
    model_seeds = list(range(args.num_models))
    seeds = list(range(args.num_seeds))

    num_models = args.num_models
    models = list()

    logits = torch.zeros((num_models, num_nodes, num_classes), device=device)
    preds = torch.zeros((num_models, num_nodes),
                        dtype=torch.int, device=device)
    confs = torch.zeros((num_models, num_nodes),
                        dtype=torch.float16, device=device)
    nodes = torch.zeros((num_models, num_nodes), dtype=torch.long)

    i = 0
    # print(now_train_mask.sum().item(), now_val_mask.sum().item())
    models.append(
        train_model(
            model_cls,
            dataset,
            o_y,
            edges,
            o_train_mask,
            o_val_mask,
            o_test_mask,
            epochs,
            device,
            model_seeds[i],
            end="\n",
        )
    )

    logits[i] = F.softmax(models[i](data.x, edges), dim=1).detach()
    l = logits[i].max(dim=1)
    preds[i] = l.indices
    confs[i] = l.values.to(device, dtype=torch.float16)
    nodes[i] = confs[i].argsort(descending=True)

    i = 1
    # print(now_train_mask.sum().item(), now_val_mask.sum().item())
    models.append(
        train_model(
            model_cls,
            dataset,
            o_y,
            edges,
            o_val_mask,
            o_train_mask,
            o_test_mask,
            epochs,
            device,
            model_seeds[i],
            end="\n",
        )
    )

    logits[i] = F.softmax(models[i](data.x, edges), dim=1).detach()
    l = logits[i].max(dim=1)
    preds[i] = l.indices
    confs[i] = l.values.to(device, dtype=torch.float16)
    nodes[i] = confs[i].argsort(descending=True)
    del models

    tna_thres = load_tna_thres(args.template, args.num_splits)
    tu_thres = load_tu_thres(args.filename)

    results = np.zeros((3, args.num_splits, args.num_seeds))
    for split_id in range(args.num_splits):
        for seed in range(args.num_seeds):
            print(
                split_id,
                seed,
                tna_thres[split_id, seed],
                tu_thres[split_id, seed],
                end=" ",
            )
            new_train_mask, new_y = enlarge(
                o_y,
                o_train_mask,
                o_test_mask,
                num_nodes,
                num_classes,
                num_models,
                nodes,
                preds,
                tna_thres[split_id, seed],
            )

            edges = torch.load(
                os.path.join(
                    "7_edges",
                    args.dataset.lower(),
                    "dropper2",
                    f"{args.model.lower()}_{args.split_id}_{0.01*tu_thres[split_id, seed]:.2f}.edges",
                )
            )
            train_acc, val_acc, test_acc = train_model_accs(
                model_cls,
                dataset,
                new_y,
                edges,
                new_train_mask,
                o_val_mask,
                o_test_mask,
                epochs,
                device,
                seed=seed,
                end="\n",
            )
            results[0, split_id, seed] = train_acc
            results[1, split_id, seed] = val_acc
            results[2, split_id, seed] = test_acc
    print(results[2].mean())
    print(results[2].var())
    # for seed_idx in range(args.num_seeds):
    #     print(" $ ", end="")
    #     train_model(
    #         model_cls,
    #         dataset,
    #         new_y,
    #         edges,
    #         new_train_mask,
    #         o_val_mask,
    #         o_test_mask,
    #         epochs,
    #         device,
    #         seed=seeds[seed_idx],
    #     )
    print()


def enlarge(
    o_y,
    o_train_mask,
    o_test_mask,
    num_nodes,
    num_classes,
    num_models,
    nodes,
    preds,
    threshold,
):
    new_y = o_y.clone()
    new_train_mask = o_train_mask.clone()
    pre_add_train_mask = torch.zeros_like(o_train_mask, dtype=torch.bool)
    num_pre_selected = num_nodes * threshold // 100
    pre_selected = nodes[:, :num_pre_selected]
    # print(f"{threshold} pre_selected: {num_pre_selected}", end=" ")
    num_added = 0
    inconsistent = 0
    num_yes = 0
    num_no = 0
    selected = pre_selected[0]
    for j in range(1, num_models):
        selected = np.intersect1d(selected, pre_selected[j])
    selected = torch.from_numpy(selected)
    # print("selected:", selected.size(0), end=" ")
    all_preds = preds[:, selected]
    for col in range(all_preds.size(1)):
        if not o_test_mask[selected[col]]:
            continue
        base_pred = all_preds[0, col]
        con_flag = True
        for row in range(1, num_models):
            now_pred = all_preds[row, col]
            # print(now_pred, base_pred)
            if now_pred.item() != base_pred.item():
                inconsistent += 1
                con_flag = False
                break
        if con_flag:
            pre_add_train_mask[selected[col]] = True
    base_preds = preds[0]
    add_c_num = list()
    for c in range(num_classes):
        add_c_num.append((base_preds[pre_add_train_mask] == c).sum().item())
    # last_add_per_c_num = add_per_c_num
    add_per_c_num = min(add_c_num)

    remain_per_c_num = [add_per_c_num for _ in range(num_classes)]

    for node in selected:
        base_pred = base_preds[node].item()
        if pre_add_train_mask[node] and remain_per_c_num[base_pred] > 0:
            new_train_mask[node] = True
            num_added += 1
            remain_per_c_num[base_pred] -= 1
            new_y[node] = base_preds[node]
            if base_pred == o_y[node].item():
                num_yes += 1
            else:
                num_no += 1
    return new_train_mask, new_y


def enlarge_train(model, optimizer, data, y, edge_index, train_mask):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(
        F.log_softmax(model(data.x, edge_index)[
                      train_mask], dim=1), y[train_mask]
    ).backward()
    optimizer.step()


def train_model_accs(
    model_cls,
    dataset,
    y,
    edges,
    train_mask,
    val_mask,
    test_mask,
    epochs,
    device,
    seed=0,
    end=" ",
):
    torch.manual_seed(seed)
    data = dataset[0]
    model = model_cls(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)
    edges = edges.to(device)
    y = y.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = test_acc = 0
    for epoch in range(epochs):
        enlarge_train(model, optimizer, data, y, edges, train_mask)
        train_acc, val_acc, tmp_test_acc = test(
            model, data, edges, train_mask, val_mask, test_mask
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
    print(epochs, "seed:", seed, train_acc, val_acc, test_acc, end=end)
    return train_acc, val_acc, test_acc


def train_model(
    model_cls,
    dataset,
    y,
    edges,
    train_mask,
    val_mask,
    test_mask,
    epochs,
    device,
    seed=0,
    end=" ",
):
    torch.manual_seed(seed)
    data = dataset[0]
    model = model_cls(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)
    edges = edges.to(device)
    y = y.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = test_acc = 0
    for epoch in range(epochs):
        enlarge_train(model, optimizer, data, y, edges, train_mask)
        train_acc, val_acc, tmp_test_acc = test(
            model, data, edges, train_mask, val_mask, test_mask
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
    print(epochs, "seed:", seed, train_acc, val_acc, test_acc, end=end)
    return model


def load_tu_thres(filename, parsed=None, num_splits=10, *args, **kwargs):
    # dim-0 Train Val Test
    # dim-1 threshold_i
    # dim-2 split
    # dim-3 seed
    # dim-4 epoch
    data = load_data(filename)
    converged_accs = data
    if not parsed:
        converged_accs = data[:, :, :, :, -1]
    if num_splits < 10:
        converged_accs = converged_accs[:, :, :num_splits, :]
    best_acc_ind = np.argmax(converged_accs, axis=1)
    print(best_acc_ind[2])
    return best_acc_ind[1]


def _load_tna_data(filename):
    NUM_SEEDS = 10
    NUM_MODELS = 2
    result = np.zeros((3, NUM_SEEDS, 101))
    lines = None
    with open(filename, "r") as f:
        lines = f.readlines()
    words = lines[0].split()
    # for i in range(3):
    #     result[i, 0] = float(words[2 + i])
    for line in lines[NUM_MODELS:]:
        words = line.split()
        if words[-1] == "improving":
            continue
        threshold = int(words[0])
        # print(threshold, end=" ")
        segments = line.split("$")
        for seg_idx in range(NUM_SEEDS):
            words = segments[seg_idx + 1].split()
            for i in range(3):
                result[i, seg_idx, threshold] = float(words[i - 3])
    return result[0], result[1], result[2]


def load_tna_thres(template, num_splits):
    NUM_SEEDS = 10
    train_accs = np.zeros((num_splits, NUM_SEEDS, 101))
    val_accs = np.zeros((num_splits, NUM_SEEDS, 101))
    test_accs = np.zeros((num_splits, NUM_SEEDS, 101))
    for i in range(num_splits):
        filename = template.replace("+0+", f"_{i}_")
        # print(filename)
        train_accs[i], val_accs[i], test_accs[i] = _load_tna_data(filename)

    print(val_accs.argmax(axis=2))
    return val_accs.argmax(axis=2)


if __name__ == "__main__":
    _main()
