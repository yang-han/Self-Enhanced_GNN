import argparse

import numpy as np
import torch

from utils import load_data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "result",
            "5_result",
            "new_5_result",
            "val_test_result",
            "test_baseline",
        ],
    )
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--rs", action="store_true")
    parser.add_argument("--parsed", action="store_true")
    return parser


def stats_test_baseline(filename, parsed=None, *args, **kwargs):
    data = load_data(filename)
    converged_test_accs = data[2, 0, :, :, -1]
    print(converged_test_accs.mean())
    report_val_accs = data[1, 0, :, :, :]
    report_val_ind = report_val_accs.argmax(axis=2)
    new_report_val_ind = torch.from_numpy(
        report_val_accs).argmax(dim=2).numpy()
    print(new_report_val_ind.shape)
    tmp_test_accs = data[2, 0, :, :, :]
    shape = tmp_test_accs.shape
    report_test_accs = (
        tmp_test_accs.reshape(shape[0] * shape[1], shape[2])
        .transpose()[report_val_ind]
        .diagonal()
    )
    print(report_test_accs.mean())
    report_test_accs = (
        tmp_test_accs.reshape(shape[0] * shape[1], shape[2])
        .transpose()[new_report_val_ind]
        .diagonal()
    )
    print(report_test_accs.mean())


def stats_val_test_result(filename, parsed=None, num_splits=10, *args, **kwargs):
    # dim-0 Train Val Test
    # dim-1 threshold_i
    # dim-2 split
    # dim-3 seed
    # dim-4 epoch
    data = load_data(filename)
    converged_accs = data
    if not parsed:
        converged_accs = data[:, :, :, :, -1]
        # tmp_val_acc_ind = torch.from_numpy(np.argmax(converged_accs[1], axis=3))
        # data = torch.from_numpy(data)
        # converged_accs = data[:, tmp_val_acc_ind].numpy()
        # # print(tmp_val_acc_ind.shape)
        # # converged_accs = data[tmp_val_acc_ind]
        # print(converged_accs.shape)
    # exit(0)
    if num_splits < 10:
        converged_accs = converged_accs[:, :, :num_splits, :]
    best_acc_ind = np.argmax(converged_accs, axis=1)
    # best_acc_ind = torch.from_numpy(converged_accs).argmax(dim=1).numpy()
    best_acc = np.max(converged_accs, axis=1)
    base_mean = converged_accs[2, 0, :, :].mean()
    base_var = converged_accs[2, 0, :, :].var()
    print(best_acc_ind)  # 3 x splits x seeds
    print(best_acc)

    best_val_thre = best_acc_ind[1].flatten()  # splits x seeds
    best_test_thre = best_acc_ind[2].flatten()

    shape = best_acc.shape

    best_acc = best_acc.reshape(shape[0], shape[1] * shape[2])
    print(best_acc.mean(1))
    print(best_acc.var(1))

    test_accs = (
        converged_accs[2]
        .reshape(converged_accs.shape[1], shape[1] * shape[2])
        .transpose()
    )
    test_accs = torch.from_numpy(test_accs)

    print("------test--------")
    best_test_thre = torch.from_numpy(best_test_thre)
    best_test_thre = test_accs[:, best_test_thre].diagonal()
    print(best_test_thre.size())
    print(best_test_thre.mean().item())
    print(best_test_thre.var().item())

    print("---------val----test-------")
    best_val_thre = torch.from_numpy(best_val_thre)
    best_val_test_acc = test_accs[:, best_val_thre].diagonal()
    print(best_val_test_acc.size())
    print(best_val_test_acc.mean().item())
    print(best_val_test_acc.var().item())

    print("-----base------")
    print(base_mean, base_var)


def stats_new_5_result(filename, parsed=None, *args, **kwargs):
    # dim-0 Train Val Test
    # dim-1 threshold_i
    # dim-2 split
    # dim-3 seed
    # dim-4 epoch
    data = load_data(filename)
    converged_accs = data
    if not parsed:
        converged_accs = data[:, :, :, :, -1]
    best_acc_ind = np.argmax(converged_accs, axis=1)
    best_acc = np.max(converged_accs, axis=1)
    print(best_acc_ind)
    print(best_acc)

    # Temp
    if not parsed:
        print(data[:, 0, 0, 0, :])

    shape = best_acc.shape
    best_acc = best_acc.reshape(shape[0], shape[1] * shape[2])
    print(best_acc.mean(1))
    print(best_acc.var(1))


def stats_5_result(filename, parsed=None, *args, **kwargs):

    # dim-0 Train Val Test
    # dim-1 threshold_i
    # dim-2 split
    # dim-3 seed
    # dim-4 epoch
    data = load_data(filename)
    converged_accs = data
    if not parsed:
        converged_accs = data[:, :, :, :, -1]
    shape = converged_accs.shape
    converged_accs = converged_accs.reshape(
        shape[0], shape[1], shape[2] * shape[3])
    mean = converged_accs.mean(axis=2)
    var = converged_accs.var(axis=2)
    best_acc_ind = mean.argmax(axis=1)
    best_acc = mean[2, best_acc_ind[2]]
    inc_acc = best_acc - mean[2, 0]
    var_best = var[2, best_acc_ind[2]]
    var_origin = var[2, 0]
    print(f"mean {mean}", f"var {var}")
    print(
        f"best_ind: {best_acc_ind[2]}, origin_acc: {mean[2, 0]:.5f}, best_acc: {best_acc:.5f}, increase: {inc_acc:.5f}, var_best: {var_best}, var_origin: {var_origin}"
    )
    print(
        f"{mean[2, 0]:.5f}({var_origin*1e4:.2f})\t  {best_acc:.5f}({var_best*1e4:.2f})+{inc_acc:.4f}_({best_acc_ind[2]})\t"
    )


def stats_result(filename, num_seeds=10, rs=False, *args, **kwargs):
    # data:
    #
    # dim-0 Train Val Test
    # dim-1 seed
    # dim-2 threshold_i
    # dim-3 epoch
    #
    # rs data:
    #
    # dim-0 Train Val Test
    # dim-1 split
    # dim-2 seed
    # dim-3 threshold_i
    # dim-4 epoch
    data = load_data(filename)
    if rs:
        converged_accs = data[:, :, :num_seeds, :, -1]
        shape = converged_accs.shape
        converged_accs = converged_accs.reshape(
            shape[0], shape[1] * shape[2], shape[3])
    else:
        converged_accs = data[:, :num_seeds, :, -1]
    mean_over_seeds = np.mean(converged_accs, axis=1)
    var_over_seeds = np.var(converged_accs, axis=1)
    print(mean_over_seeds)
    print(mean_over_seeds.max(axis=1), mean_over_seeds.argmax(axis=1))
    print(var_over_seeds)


def stats_baseline(filename, num_seeds=100, rs=False, *args, **kwargs):
    # data:
    #
    # dim-0 Train Val Test
    # dim-1 seed
    # dim-2 epoch
    #
    # rs data:
    #
    # dim-0 Train Val Test
    # dim-1 split
    # dim-2 seed
    # dim-3 epoch
    data = load_data(filename)

    if rs:
        converged_accs = data[:, :, :num_seeds, -1]
        shape = converged_accs.shape
        converged_accs = converged_accs.reshape(shape[0], shape[1] * shape[2])
    else:
        converged_accs = data[:, :num_seeds, -1]

    mean_converged_accs = np.mean(converged_accs, axis=1)
    var_converged_accs = np.var(converged_accs, axis=1)
    print(mean_converged_accs)
    print(var_converged_accs)


def main():
    parser = create_parser()
    args = parser.parse_args()
    func = globals()[f"stats_{args.mode}"]
    func(
        args.filename,
        num_seeds=args.num_seeds,
        rs=args.rs,
        parsed=args.parsed,
        num_splits=args.num_splits,
    )


if __name__ == "__main__":
    main()
