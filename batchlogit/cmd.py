import click
import click_log
import torch

from .api import METHODS, logit_runner_by_name

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

            for _ in range(1024):
                yield [
                    torch.as_tensor(t, dtype=torch.float32)
                    for t in make_classification(40, 10, random_state=rng)
                ]
        else:
            from cuml.datasets import make_classification
            from cupy.random import RandomState

            rng = RandomState(42)
            device_obj = torch.device("cuda")

            for _ in range(1024):
                yield [
                    torch.as_tensor(t, dtype=torch.float32, device=device_obj)
                    for t in make_classification(40, 10, random_state=rng)
                ]

    runner = logit_runner_by_name(method, n_jobs)

    def run_once(timer_msg, outfn=None):
        with Timer(timer_msg):
            fit_n_iters_list, weights_list, biases_list = runner(prob_it())
        if outfn:
            weights = torch.vstack(weights_list)
            biases = torch.vstack(biases_list)[:, 0]
            torch.save((fit_n_iters_list, weights, biases), outfn)

    try:
        run_once("Cold start 1024")
        run_once("Warm start 1024", outfn)
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
