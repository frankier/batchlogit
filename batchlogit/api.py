from .methods import (
    lr_many_pytorch_lgfbs,
    lr_one_cuml,
    lr_one_nlesc_dirac_lbgfs,
    lr_one_pytorch_hjmshi_lgfbs,
    lr_one_pytorch_lgfbs,
    lr_one_skl,
)
from .runners import ChunkRunner, JoblibRunner, PyTorchMpPool, SerialRunner

METHODS = [
    "cuml_joblib_serial",
    "cuml_joblib_threading",
    "cuml_joblib_loky",
    "pytorch_lgfbs_chunk",
    "pytorch_lgfbs_mp",
    "pytorch_lgfbs_serial",
    "pytorch_nlesc_dirac_lbgfs_serial",
    "pytorch_hjmshi_lgfbs_serial",
    "skl_joblib_loky",
    "skl_serial",
]


def logit_runner_by_name(name, n_jobs=None):
    if name == "cuml_joblib_serial":
        return SerialRunner(lr_one_cuml)
    elif name == "cuml_joblib_threading":
        return JoblibRunner(lr_one_cuml, n_jobs=n_jobs, backend="threading")
    elif name == "cuml_joblib_loky":
        return JoblibRunner(lr_one_cuml, n_jobs=n_jobs)
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
