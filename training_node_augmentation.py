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
from utils import EPOCHS_CONFIG, load_dataset, load_split, test, train


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--model", type=str, default="GCN")
    # parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-seeds", type=int, default=10)
    # parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--splits-dir", type=str, default="splits")
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--num-models", type=int, default=2)
    parser.add_argument("--s-thres", type=int, default=0)
    parser.add_argument("--e-thres", type=int, default=101)
    parser.add_argument("--j-thres", type=int, default=1)
    # parser.add_argument("--enlarge", type=float, default=0.5)
    parser.add_argument("--verbose", "-v", action="store_true")
    # parser.add_argument("--output", type=str)

    return parser


# def generate_train_val(seen_mask, y, num_classes, seed=0):
#     torch.manual_seed(seed)
#     seen_index = seen_mask.nonzero().flatten()
#     new_train_mask = torch.zeros_like(seen_mask, dtype=torch.bool)
#     new_val_mask = torch.zeros_like(seen_mask, dtype=torch.bool)
#     num_seen = seen_mask.sum()
#     per_c_num_seen = num_seen // num_classes
#     for c in range(num_classes):
#         c_idx = seen_index[(y[seen_mask] == c).nonzero().flatten()]
#         perm = torch.randperm(per_c_num_seen)
#         # print(perm, perm.size())
#         train_idx = c_idx[perm[: per_c_num_seen // 2]]
#         val_idx = c_idx[perm[per_c_num_seen // 2 :]]
#         new_train_mask[train_idx] = True
#         new_val_mask[val_idx] = True
#     return new_train_mask, new_val_mask


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

    add_per_c_num = -1
    for i in range(args.s_thres, args.e_thres, args.j_thres):
        new_y = o_y.clone()
        new_train_mask = o_train_mask.clone()
        pre_add_train_mask = torch.zeros_like(o_train_mask, dtype=torch.bool)
        num_pre_selected = num_nodes * i // 100
        pre_selected = nodes[:, :num_pre_selected]
        print(f"{i} pre_selected: {num_pre_selected}", end=" ")
        num_added = 0
        inconsistent = 0
        num_yes = 0
        num_no = 0
        selected = pre_selected[0]
        for j in range(1, num_models):
            selected = np.intersect1d(selected, pre_selected[j])
        selected = torch.from_numpy(selected)
        print("selected:", selected.size(0), end=" ")
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
            add_c_num.append(
                (base_preds[pre_add_train_mask] == c).sum().item())
        last_add_per_c_num = add_per_c_num
        add_per_c_num = min(add_c_num)
        if add_per_c_num == 0 and i != args.s_thres:
            print(add_c_num, "skipping for not improving")
            continue

        if last_add_per_c_num == add_per_c_num:
            print(add_c_num, "skipping for not improving")
            continue

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
        print(
            f"num_added: {num_added} inconsistent: {inconsistent} per_c: {add_c_num} | add_per_c: {add_per_c_num} num_yes/no: {num_yes} {num_no}",
            end=" ",
        )
        for seed_idx in range(args.num_seeds):
            print(" $ ", end="")
            train_model(
                model_cls,
                dataset,
                new_y,
                edges,
                new_train_mask,
                o_val_mask,
                o_test_mask,
                epochs,
                device,
                seed=seeds[seed_idx],
            )
        print()


def enlarge_train(model, optimizer, data, y, edge_index, train_mask):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(
        F.log_softmax(model(data.x, edge_index)[
                      train_mask], dim=1), y[train_mask]
    ).backward()
    optimizer.step()


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


if __name__ == "__main__":
    _main()
