from .methods import (
    lr_many_pytorch_lbfgs,
    lr_one_cuml,
    lr_one_nlesc_dirac_lbgfs,
    lr_one_pytorch_hjmshi_lbfgs,
    lr_one_pytorch_lbfgs,
    lr_one_skl,
)
from .runners import ChunkRunner, CopyWrapper, JoblibRunner, PyTorchMpPool, SerialRunner

METHODS = [
    "cuml_joblib_serial",
    "cuml_joblib_threading",
    "cuml_joblib_loky",
    "pytorch_lbfgs_chunk",
    "pytorch_lbfgs_mp",
    "pytorch_lbfgs_serial",
    "pytorch_nlesc_dirac_lbgfs_serial",
    "pytorch_hjmshi_lbfgs_serial",
    "skl_joblib_loky",
    "skl_serial",
]


def logit_runner_by_name(name, n_jobs=None, device=None):
    if name == "cuml_joblib_serial":
        return CopyWrapper(SerialRunner(lr_one_cuml), device or "cuda")
    elif name == "cuml_joblib_threading":
        return CopyWrapper(
            JoblibRunner(lr_one_cuml, n_jobs=n_jobs, backend="threading"),
            device or "cuda",
        )
    elif name == "cuml_joblib_loky":
        return CopyWrapper(JoblibRunner(lr_one_cuml, n_jobs=n_jobs), device or "cuda")
    elif name == "pytorch_lbfgs_chunk":
        return ChunkRunner(lr_many_pytorch_lbfgs, chunk_size=n_jobs)
    elif name == "pytorch_lbfgs_mp":
        return PyTorchMpPool(lr_one_pytorch_lbfgs, processes=n_jobs)
    elif name == "pytorch_lbfgs_serial":
        return SerialRunner(lr_one_pytorch_lbfgs)
    elif name == "pytorch_nlesc_dirac_lbgfs_serial":
        return SerialRunner(lr_one_nlesc_dirac_lbgfs)
    elif name == "pytorch_hjmshi_lbfgs_serial":
        return SerialRunner(lr_one_pytorch_hjmshi_lbfgs)
    elif name == "skl_joblib_loky":
        return CopyWrapper(JoblibRunner(lr_one_skl, n_jobs=n_jobs), device or "cpu")
    else:
        assert name == "skl_serial"
        return CopyWrapper(SerialRunner(lr_one_skl), device or "cpu")
