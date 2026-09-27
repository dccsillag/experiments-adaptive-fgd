#!/usr/bin/env bash

RUN_FOLDER="$1"

# compute txt
python src/functional_radiance/eval/eval_nerf.py -p "$1"

parallel -k \
    python src/functional_radiance/eval/eval_nerf_per_epoch.py -p {} \
    ::: "$1"/testset_*  > "$1"/test-eval.jsonl

cat "$1/log.txt" |\
    awk '/Step:/ {t += $6 ; print "[" int($2) ", " t "," ($4 +0) "]"}' |\
    jq -c '{"epoch":.[0], "time": .[1], "train_mse":.[2]}' >\
    "$1/times.jsonl"
