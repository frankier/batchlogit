#!/bin/bash


run() {
  echo "$*"
  python -m batchlogit.cmd $*
}

echo " ** CPU ** "
run cuml_joblib_serial
run --n-jobs=10 cuml_joblib_threading
run --n-jobs=10 cuml_joblib_loky
run pytorch_lgfbs_serial
run --n-jobs=10 pytorch_lgfbs_chunk
run pytorch_nlesc_dirac_lbgfs_serial
run pytorch_hjmshi_lgfbs_serial
run --n-jobs=10 skl_joblib_loky
run skl_serial

echo " ** GPU ** "
run --device gpu cuml_joblib_serial cuml.pt
run --device gpu --n-jobs=10 cuml_joblib_threading
run --device gpu --n-jobs=10 cuml_joblib_loky
run --device gpu pytorch_lgfbs_serial lgfbs_serial.pt
run --device gpu --n-jobs=10 pytorch_lgfbs_chunk lgfbs_chunk.pt
run --device gpu pytorch_nlesc_dirac_lbgfs_serial nlesc_dirac_lbgfs_serial.pt
run --device gpu pytorch_hjmshi_lgfbs_serial hjmshi_lgfbs_serial.pt
run --device gpu --n-jobs=10 skl_joblib_loky
run --device gpu skl_serial skl.pt
