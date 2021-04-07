import sys

import torch

cpu_dev = torch.device("cpu")

for idx, fn in enumerate(sys.argv[1:]):
    fit_n_iters_list, weights, biases = torch.load(fn, cpu_dev)
    print(fn)
    print(weights)
    print(biases)

print()
print("***")
print()

for idx, fn in enumerate(sys.argv[1:]):
    fit_n_iters_list, weights, biases = torch.load(fn, cpu_dev)
    for other_idx, other_fn in enumerate(sys.argv[1:]):
        if other_idx >= idx:
            continue
        other_fit_n_iters_list, other_weights, other_biases = torch.load(
            other_fn, cpu_dev
        )
        print(f"Difference between {fn} and {other_fn}")
        weights_abs_diff = (weights - other_weights).abs()
        biases_abs_diff = (biases - other_biases).abs()
        print("Max", weights_abs_diff.max(), biases_abs_diff.max())
        print("Sums", weights_abs_diff.sum(), biases_abs_diff.sum())
        print("Means", weights_abs_diff.mean(), biases_abs_diff.mean())
