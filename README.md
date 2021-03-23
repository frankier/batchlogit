# batchlogit

This repository is an experiment/benchmark for training in parallel many
logistic regression models.

## Installing

Install using `conda env create -f environment.yml` or the [container image for
Docker/Singularity](https://github.com/users/frankier/packages/container/package/batchlogit).

## Systems

There are several systems. They all attempt to implement something mostly
equivalent to the L2 regularised implementation in scikit-learn.

 * Existing implementation in CuML
 * Existing implementation in Scikit-learn
 * Different implementations based on PyTorch and LBFGS optimisers
   * Standard implementation using torch.optim.LBFGS
   * "Chunked" implementation which optimises the concatenation of many models
     at the same time
   * Implementation using LBFGS from here:
     https://github.com/hjmshi/PyTorch-LBFGS
   * Implementation using LBFGS from here:
     https://github.com/nlesc-dirac/pytorch

They are parallelized in various ways:

 * Standard serial approach
 * Joblib based parallelization
   * Threading
   * Loky based multiprocessing
 * PyTorch multiprocessing based parallelization since this allows CUDA tensors
   to be safely shared

There is some effort to configure optimizers to be more or less equivalent in
terms of things like number of iterations and stopping criteria, but it's not
always possible since there are not always equivalent settings.

## Benchmark

The benchmark makes 1024 problems each with 40 points in 10-dimensional feature
space. There are two different setups. In the CPU setup, the data begins in CPU
memory. In this setup, it is copied to the GPU and back to run on CuML, and
everything else runs on the CPU. In the GPU setup, the data begins in GPU
memory. In this setup, it is copied to the CPU to run on scikit-learn, and
everything else runs on the GPU.

The benchmark was run on [1/4 of a CSC Puhti GPU
machine](https://docs.csc.fi/computing/systems-puhti/). That means ~10 cores,
90GiB memory and 1 V100 GPU.

This benchmark should in no way be taken as definitive, since it has some
obvious caveats from the starts, such as the fact other things, like the
creation of the benchmark problems are also being measured. PRs and alternative
numbers/interpretations welcome.

### Results

```
 ** CPU ** 
cuml_joblib_serial
init
[Time] Cold start 1024 consumes 26.8370 s
[Time] Warm start 1024 consumes 6.1292 s
--n-jobs=10 cuml_joblib_threading
init
[Time] Cold start 1024 consumes 12.9894 s
[Time] Warm start 1024 consumes 7.1528 s
--n-jobs=10 cuml_joblib_loky
[Time] Cold start 1024 consumes 29.0066 s
[Time] Warm start 1024 consumes 19.2028 s
pytorch_lgfbs_serial
[Time] Cold start 1024 consumes 10.9807 s
[Time] Warm start 1024 consumes 10.5207 s
--n-jobs=10 pytorch_lgfbs_chunk
[Time] Cold start 1024 consumes 5.2894 s
[Time] Warm start 1024 consumes 5.1270 s
pytorch_nlesc_dirac_lbgfs_serial
[Time] Cold start 1024 consumes 23.1817 s
[Time] Warm start 1024 consumes 22.7755 s
pytorch_hjmshi_lgfbs_serial
ERROR
--n-jobs=10 skl_joblib_loky
[Time] Cold start 1024 consumes 1.6706 s
[Time] Warm start 1024 consumes 0.6998 s
skl_serial
[Time] Cold start 1024 consumes 2.5684 s
[Time] Warm start 1024 consumes 2.2841 s

 ** GPU ** 
--device gpu cuml_joblib_serial cuml.pt
init
[Time] Cold start 1024 consumes 19.1638 s
[Time] Warm start 1024 consumes 9.3143 s
--device gpu --n-jobs=10 cuml_joblib_threading
init
[Time] Cold start 1024 consumes 17.7480 s
[Time] Warm start 1024 consumes 10.1351 s
--device gpu --n-jobs=10 cuml_joblib_loky
[Time] Cold start 1024 consumes 34.2052 s
[Time] Warm start 1024 consumes 20.3826 s
--device gpu pytorch_lgfbs_serial lgfbs_serial.pt
[Time] Cold start 1024 consumes 37.5870 s
[Time] Warm start 1024 consumes 31.2701 s
--device gpu --n-jobs=10 pytorch_lgfbs_chunk lgfbs_chunk.pt
[Time] Cold start 1024 consumes 20.6648 s
[Time] Warm start 1024 consumes 14.5042 s
--device gpu pytorch_nlesc_dirac_lbgfs_serial nlesc_dirac_lbgfs_serial.pt
[Time] Cold start 1024 consumes 65.8933 s
[Time] Warm start 1024 consumes 59.7671 s
--device gpu pytorch_hjmshi_lgfbs_serial hjmshi_lgfbs_serial.pt
ERROR
--device gpu --n-jobs=10 skl_joblib_loky
[Time] Cold start 1024 consumes 11.7210 s
[Time] Warm start 1024 consumes 4.3246 s
--device gpu skl_serial skl.pt
[Time] Cold start 1024 consumes 12.5219 s
[Time] Warm start 1024 consumes 6.2105 s
```

### Analysis

Scikit-learn beats all. Even if your problems start out on the GPU, if you have
many tiny problems like this, it's actually faster to copy them to the CPU, run
scikit-learn on them, and copy them back. One clear problem for CuML is that it
doesn't seem to be able to take advantage of coarse-grained parallelism.

### Comparison of outputs

The "chunked" LBFGS produces significantly different results to the others. It
is not actually equivalent to the others. Perhaps with more work, this approach
of running it in lockstep could be made equivalent or near-equivalent, so it
has been left in for this reason.

## Vendorised code

This repository contains vendorised code from the following repositories:

 * https://github.com/hjmshi/PyTorch-LBFGS
 * https://github.com/nlesc-dirac/pytorch
