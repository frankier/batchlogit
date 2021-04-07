import torch
from joblib import Parallel, delayed
from more_itertools import chunked


class JoblibRunner:
    def __init__(self, logistic_regression, **kwargs):
        self.logistic_regression = logistic_regression
        self._parallel_pool = Parallel(**kwargs)
        self.parallel = self._parallel_pool.__enter__()

    def stop(self):
        if self.parallel is not None:
            self._parallel_pool.__exit__(None, None, None)

    def __call__(self, prob_it, **kwargs):
        logits_delayed = []
        for feats_support, gt_support in prob_it:
            logits_delayed.append(
                delayed(self.logistic_regression)(feats_support, gt_support, **kwargs)
            )
        fit_n_iters_list = []
        weights = []
        biases = []
        for result in self.parallel(logits_delayed):
            if result is None:
                continue
            weight, bias, lr_fit_n_iters = result
            fit_n_iters_list.append(lr_fit_n_iters)
            weights.append(weight)
            biases.append(bias)
        return fit_n_iters_list, weights, biases


class PyTorchMpPool:
    def __init__(self, logistic_regression, **kwargs):
        from torch import multiprocessing

        # multiprocessing.set_sharing_strategy("file_system")
        self.logistic_regression = logistic_regression
        ctx = multiprocessing.get_context("spawn")
        self._pool = ctx.Pool(**kwargs)
        self.parallel = self._pool.__enter__()
        self.kwargs = {}

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

        res = self.logistic_regression(*xy, **self.kwargs)
        empty_cache()
        return res

    def __call__(self, prob_it, **kwargs):
        fit_n_iters_list = []
        weights = []
        biases = []
        self.kwargs = kwargs
        results = self.parallel.imap_unordered(self._lr_one, prob_it)
        for result in results:
            if result is None:
                continue
            weight, bias, lr_fit_n_iters = result
            fit_n_iters_list.append(lr_fit_n_iters)
            weights.append(weight)
            biases.append(bias)
        return fit_n_iters_list, weights, biases


class SerialRunner:
    def __init__(self, logistic_regression):
        self.logistic_regression = logistic_regression

    def stop(self):
        pass

    def __call__(self, prob_it, **kwargs):
        fit_n_iters_list = []
        weights = []
        biases = []
        for feats_support, gt_support in prob_it:
            result = self.logistic_regression(feats_support, gt_support, **kwargs)
            if result is None:
                continue
            weight, bias, lr_fit_n_iters = result
            fit_n_iters_list.append(lr_fit_n_iters)
            weights.append(weight)
            biases.append(bias)
        return fit_n_iters_list, weights, biases


class ChunkRunner:
    def __init__(self, logistic_regression, chunk_size=None):
        self.logistic_regression = logistic_regression
        if chunk_size is None:
            chunk_size = 4
        self.chunk_size = chunk_size

    def stop(self):
        pass

    def __call__(self, prob_it, **kwargs):
        fit_n_iters_list = []
        weights = []
        biases = []
        for chunk in chunked(prob_it, self.chunk_size):
            feats_support, gt_support = zip(*chunk)
            weight, bias, lr_fit_n_iters = self.logistic_regression(
                torch.stack(feats_support, axis=0),
                torch.stack(gt_support, axis=0),
                **kwargs,
            )
            fit_n_iters_list.append(lr_fit_n_iters)
            weights.append(weight)
            biases.append(bias)
        return fit_n_iters_list, weights, biases


class CopyWrapper:
    def __init__(self, runner, device_name):
        self.runner = runner
        self.device = torch.device(device_name)

    def stop(self):
        self.runner.stop()

    def __call__(self, prob_it, **kwargs):
        initial_device = None

        def wrapped_it():
            nonlocal initial_device
            for feats_support, gt_support in prob_it:
                if initial_device is None:
                    initial_device = feats_support.device
                yield feats_support.to(self.device), gt_support.to(self.device)

        fit_n_iters_list, weights, biases = self.runner(wrapped_it(), **kwargs)
        assert initial_device is not None
        return (
            fit_n_iters_list,
            [weight.to(initial_device) for weight in weights],
            [bias.to(initial_device) for bias in biases],
        )
