#!/bin/bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
    ftp_proxy FTP_PROXY no_proxy NO_PROXY

export http_proxy=
export https_proxy=
export HTTP_PROXY=
export HTTPS_PROXY=
export all_proxy=
export ALL_PROXY=

python scripts/run_experiment.py configs/bert_baseline.yaml