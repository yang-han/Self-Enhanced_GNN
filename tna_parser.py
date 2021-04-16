import argparse

import numpy as np

# NUM_SPLITS = 10
NUM_MODELS = 2
NUM_SEEDS = 10


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, required=True)
    # parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--num-splits", type=int, default=10)
    return parser


def parse(filename):
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
        print(threshold, end=" ")
        segments = line.split("$")
        for seg_idx in range(NUM_SEEDS):
            words = segments[seg_idx + 1].split()
            for i in range(3):
                result[i, seg_idx, threshold] = float(words[i - 3])
    return result[0], result[1], result[2]


def main(_args=None):
    args = create_parser().parse_args(_args)
    num_splits = args.num_splits

    train_accs = np.zeros((num_splits, NUM_SEEDS, 101))
    val_accs = np.zeros((num_splits, NUM_SEEDS, 101))
    test_accs = np.zeros((num_splits, NUM_SEEDS, 101))
    for i in range(num_splits):
        # if i==3 :
        #     continue
        filename = args.template.replace("+0+", f"_{i}_")
        print(filename)
        train_accs[i], val_accs[i], test_accs[i] = parse(filename)

    print(test_accs[:, :, 0])
    print(val_accs[:, :, 0])

    test_accs = test_accs.reshape((num_splits * NUM_SEEDS, 101))
    val_accs = val_accs.reshape((num_splits * NUM_SEEDS, 101))

    print("---base---")
    print(test_accs[:, 0].mean())
    print(test_accs[:, 0].var())

    print("---best-val-test---")
    threshold = val_accs.argmax(axis=1)
    print(threshold)
    best_val_test_accs = test_accs[:, threshold].diagonal()
    print(best_val_test_accs.mean())
    print(best_val_test_accs.var())


if __name__ == "__main__":
    main()
