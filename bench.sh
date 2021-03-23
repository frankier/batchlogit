run() {
  echo "$*"
  python -m batchlogit.cmd $*
}

echo " ** CPU ** "
run --n-jobs=1 cuml_joblib_threading
run --n-jobs=10 cuml_joblib_threading
run pytorch_lgfbs_serial
run --n-jobs=10 pytorch_lgfbs_chunk
run --n-jobs=10 skl_joblib_loky
run skl_serial

echo " ** GPU ** "
run --device gpu --n-jobs=1 cuml_joblib_threading
run --device gpu --n-jobs=10 cuml_joblib_threading
run --device gpu pytorch_lgfbs_serial
run --device gpu --n-jobs=10 pytorch_lgfbs_chunk
run --device gpu --n-jobs=10 skl_joblib_loky
run --device gpu skl_serial
