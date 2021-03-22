import click
import click_log
import torch
from joblib import Parallel, delayed
from more_itertools import chunked
from torch import nn

click_log.basic_config()


def lr_one_skl(x: torch.tensor, y: torch.tensor):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    initial_device = x.device
    x = x.detach()
    model = LogisticRegression()
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ConvergenceWarning)
        model = model.fit(x, y)
    weight = model.coef_
    bias = model.intercept_
    weight = torch.as_tensor(weight, dtype=torch.float32, device=initial_device)
    bias = torch.as_tensor(bias, dtype=torch.float32, device=initial_device)
    n_iter = model.n_iter_
    return weight, bias, n_iter


def lr_one_cuml(x: torch.tensor, y: torch.tensor):
    from cuml.linear_model import LogisticRegression

    initial_device = x.device
    x = x.detach()
    y = y.float()
    model = LogisticRegression()
    model = model.fit(x, y)
    weight = model.coef_
    bias = model.intercept_
    weight = torch.as_tensor(weight, device=initial_device).float().t()
    bias = torch.as_tensor(bias, device=initial_device).float()
    n_iter = model.solver_model.num_iters
    return weight, bias, n_iter


class StackedRegLogitModel(torch.nn.Module):
    def __init__(self, chunk_size, input_dim, output_dim, C=1):
        super(StackedRegLogitModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim) for _ in range(chunk_size)]
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.C = C

    def forward(self, x):
        y_all = [linear(vec) for linear, vec in zip(self.linears, x)]
        return torch.stack(y_all, axis=0)

    def forward_loss(self, x, y):
        y_hat_logit = self(x)
        model_loss = self.loss_fn(y_hat_logit[:, :, 0], y)
        # Don't regularize bias to match scikit-learn/cuml
        l2_reg = 0.5 * sum((torch.mm(linear.weight, linear.weight.t()) for linear in self.linears))
        return model_loss + l2_reg / self.C


class RegLogitModel(nn.Module):
    def __init__(self, input_dim, output_dim, C=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.C = C

    def forward(self, x):
        return self.linear(x)

    def forward_loss(self, x, y):
        y_hat_logit = self(x)
        model_loss = self.loss_fn(y_hat_logit[:, 0], y)
        # Don't regularize bias to match scikit-learn/cuml
        l2_reg = 0.5 * torch.mm(self.linear.weight, self.linear.weight.t())
        return model_loss + l2_reg / self.C


def lr_many_pytorch_lgfbs(
    x,
    y,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-4,
    C=1,
):
    from torch.optim import LBFGS

    model = StackedRegLogitModel(x.shape[0], x.shape[-1], 1, C=C)
    optimizer = LBFGS(
        model.parameters(),
        lr=1,
        history_size=history_size,
        max_iter=max_iter,
        # XXX: Cannot pass max_ls to strong_wolfe
        line_search_fn="strong_wolfe",
        tolerance_change=0,
        tolerance_grad=tol,
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss = model.forward_loss(x_var, y_var)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    weights = []
    biases = []
    for linear in model.linears:
        weights.append(linear.weight.detach())
        biases.append(linear.bias.detach())
    return (torch.stack(weights, axis=0), torch.stack(biases, axis=0), state["n_iter"])


def lr_one_pytorch_lgfbs(
    x: torch.tensor,
    y: torch.tensor,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-4,
    C=1,
):
    from torch.optim import LBFGS

    model = RegLogitModel(x.shape[-1], 1, C=C)
    optimizer = LBFGS(
        model.parameters(),
        lr=1,
        history_size=history_size,
        max_iter=max_iter,
        # XXX: Cannot pass max_ls to strong_wolfe
        line_search_fn="strong_wolfe",
        tolerance_change=0,
        tolerance_grad=tol,
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss = model.forward_loss(x_var, y_var)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    return (model.linear.weight.detach(), model.linear.bias.detach(), state["n_iter"])


def lr_one_nlesc_dirac_lbgfs(
    x: torch.tensor,
    y: torch.tensor,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-3,
    C=1,
):
    from .nlesc_dirac_lbgfs import LBFGSNew

    model = RegLogitModel(x.shape[-1], 1, C=C)
    optimizer = LBFGSNew(
        model.parameters(),
        lr=1,
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=tol,
        tolerance_change=0,
        line_search_fn=True,
        batch_mode=False,
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss = model.forward_loss(x_var, y_var)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    return (model.linear.weight.detach(), model.linear.bias.detach(), state["n_iter"])


def lr_one_pytorch_hjmshi_lgfbs(
    x: torch.tensor,
    y: torch.tensor,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-4,
    C=1,
):
    print("XXX: This seems to be broken currently.")
    # XXX: Currently broken
    from .hjmshi_lbfgs import FullBatchLBFGS

    model = RegLogitModel(x.shape[-1], 1, C=C)
    optimizer = FullBatchLBFGS(
        model.parameters(),
        lr=1,
        history_size=history_size,
        line_search="Wolfe",
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        loss = model.forward_loss(x_var, y_var)
        return loss

    loss = closure()
    n_iter = 1
    while 1:
        options = {
            "closure": closure,
            "current_loss": loss,
            "max_ls": max_ls,
        }
        (
            loss,
            grad,
            lr,
            backtracks,
            clos_evals,
            grad_evals,
            desc_dir,
            fail,
        ) = optimizer.step(options=options)

        grad_max = grad.abs().max()

        if fail:
            raise RuntimeError("Optimizer failure in lr_one_pytorch_hjmshi_lgfbs", fail)
        elif torch.isnan(loss):
            raise RuntimeError("NaN loss in lr_one_pytorch_hjmshi_lgfbs")
        elif grad_max < tol or n_iter == max_iter:
            return (model.linear.weight.detach(), model.linear.bias.detach(), n_iter)
        n_iter += 1


class JoblibRunner:
    def __init__(self, logistic_regression, **kwargs):
        self.logistic_regression = logistic_regression
        self._parallel_pool = Parallel(**kwargs)
        self.parallel = self._parallel_pool.__enter__()

    def stop(self):
        if self.parallel is not None:
            self._parallel_pool.__exit__(None, None, None)

    def __call__(self, prob_it):
        logits_delayed = []
        for feats_support, gt_support in prob_it:
            logits_delayed.append(
                delayed(self.logistic_regression)(feats_support, gt_support)
            )
        max_lr_fit_n_iters = 0
        weights = []
        biases = []
        for weight, bias, lr_fit_n_iters in self.parallel(logits_delayed):
            if lr_fit_n_iters > max_lr_fit_n_iters:
                max_lr_fit_n_iters = lr_fit_n_iters
            weights.append(weight)
            biases.append(bias)
        return max_lr_fit_n_iters, weights, biases


class PyTorchMpPool:
    def __init__(self, logistic_regression, **kwargs):
        from torch import multiprocessing

        # multiprocessing.set_sharing_strategy("file_system")
        self.logistic_regression = logistic_regression
        ctx = multiprocessing.get_context("spawn")
        self._pool = ctx.Pool(**kwargs)
        self.parallel = self._pool.__enter__()

    def __getstate__(self):
        """
        This is the state we want to get pickled to be passed to the worker
        process.
        """
        return {
            "logistic_regression": self.logistic_regression,
        }

    def stop(self):
        if self.parallel is not None:
            self._parallel_pool.__exit__(None, None, None)

    def _lr_one(self, xy):
        from torch.cuda import empty_cache

        res = self.logistic_regression(*xy)
        empty_cache()
        return res

    def __call__(self, prob_it):
        max_lr_fit_n_iters = 0
        weights = []
        biases = []
        results = self.parallel.imap_unordered(self._lr_one, prob_it)
        for weight, bias, lr_fit_n_iters in results:
            if lr_fit_n_iters > max_lr_fit_n_iters:
                max_lr_fit_n_iters = lr_fit_n_iters
            weights.append(weight)
            biases.append(bias)
        return max_lr_fit_n_iters, weights, biases


class SerialRunner:
    def __init__(self, logistic_regression):
        self.logistic_regression = logistic_regression

    def stop(self):
        pass

    def __call__(self, prob_it):
        max_lr_fit_n_iters = 0
        weights = []
        biases = []
        for feats_support, gt_support in prob_it:
            weight, bias, lr_fit_n_iters = self.logistic_regression(
                feats_support, gt_support
            )
            if lr_fit_n_iters > max_lr_fit_n_iters:
                max_lr_fit_n_iters = lr_fit_n_iters
            weights.append(weight)
            biases.append(bias)
        return max_lr_fit_n_iters, weights, biases


class ChunkRunner:
    def __init__(self, logistic_regression, chunk_size=None):
        self.logistic_regression = logistic_regression
        if chunk_size is None:
            chunk_size = 4
        self.chunk_size = chunk_size

    def stop(self):
        pass

    def __call__(self, prob_it):
        max_lr_fit_n_iters = 0
        weights = []
        biases = []
        for chunk in chunked(prob_it, self.chunk_size):
            feats_support, gt_support = zip(*chunk)
            weight, bias, lr_fit_n_iters = self.logistic_regression(
                torch.stack(feats_support, axis=0), torch.stack(gt_support, axis=0)
            )
            if lr_fit_n_iters > max_lr_fit_n_iters:
                max_lr_fit_n_iters = lr_fit_n_iters
            weights.append(weight)
            biases.append(bias)
        return max_lr_fit_n_iters, weights, biases


def logit_runner_by_name(name, n_jobs=None):
    if name == "cuml_joblib_threading":
        return JoblibRunner(lr_one_cuml, n_jobs=n_jobs, backend="threading")
    elif name == "pytorch_lgfbs_chunk":
        return ChunkRunner(lr_many_pytorch_lgfbs, chunk_size=n_jobs)
    elif name == "pytorch_lgfbs_mp":
        return PyTorchMpPool(lr_one_pytorch_lgfbs, processes=n_jobs)
    elif name == "pytorch_lgfbs_serial":
        return SerialRunner(lr_one_pytorch_lgfbs)
    elif name == "pytorch_nlesc_dirac_lbgfs_serial":
        return SerialRunner(lr_one_nlesc_dirac_lbgfs)
    elif name == "pytorch_hjmshi_lgfbs_serial":
        return SerialRunner(lr_one_pytorch_hjmshi_lgfbs)
    elif name == "skl_joblib_loky":
        return JoblibRunner(lr_one_skl, n_jobs=n_jobs)
    else:
        assert name == "skl_serial"
        return SerialRunner(lr_one_skl)


@click.command()
@click_log.simple_verbosity_option()
@click.argument("method")
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
            for _ in range(1024):
                yield [
                    torch.as_tensor(t, dtype=torch.float32, device=torch.device('cuda'))
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

    run_once("Cold start 1024")
    run_once("Warm start 1024", outfn)


if __name__ == "__main__":
    main()
