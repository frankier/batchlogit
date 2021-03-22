import click
import click_log
import torch
from joblib import Parallel, delayed
from more_itertools import chunked
from torch import nn

from .api import METHODS

click_log.basic_config()


@click.command()
@click_log.simple_verbosity_option()
@click.argument("method", type=click.Choice(METHODS))
@click.argument("outfn", required=False)
@click.option("--n-jobs", type=int)
@click.option("--device", default="cpu", type=click.Choice(["cpu", "gpu"]))
def main(method, outfn, n_jobs, device):
    from .utils import Timer

    def prob_it():
        if device == "cpu":
            from numpy.random import RandomState
            from sklearn.datasets import make_classification

            rng = RandomState(42)

            def transform_tensor(t):
                res = torch.as_tensor(t, dtype=torch.float32)
                if method.startswith("cuml_"):
                    return res.cuda()
                else:
                    return res

            for _ in range(1024):
                yield [
                    transform_tensor(t)
                    for t in make_classification(40, 10, random_state=rng)
                ]
        else:
            from cuml.datasets import make_classification
            from cupy.random import RandomState

            rng = RandomState(42)
            device_obj = torch.device("cuda")

            def transform_tensor(t):
                res = torch.as_tensor(t, dtype=torch.float32, device=device_obj)
                if method.startswith("skl_"):
                    return res.cpu()
                else:
                    return res

            for _ in range(1024):
                yield [
                    transform_tensor(t)
                    for t in make_classification(40, 10, random_state=rng)
                ]

    runner = logit_runner_by_name(method, n_jobs)

    def run_once(timer_msg, outfn=None):
        with Timer(timer_msg):
            max_lr_fit_n_iters, weights_list, biases_list = runner(prob_it())
        if outfn:
            weights = torch.vstack(weights_list)
            biases = torch.vstack(biases_list)[:, 0]
            torch.save((max_lr_fit_n_iters, weights, biases), outfn)

    try:
        run_once("Cold start 1024")
        run_once("Warm start 1024", outfn)
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
