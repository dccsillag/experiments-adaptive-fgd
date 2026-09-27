#!/bin/sh -ex

mkdir -p out-wave-equation-ours
python src/wave-equation/sandbox_ours.py -o out-wave-equation-ours
mkdir -p out-wave-equation-nn
python src/wave-equation/sandbox_nn.py -o out-wave-equation-nn
mkdir -p out-wave-equation-gt
python src/wave-equation/sandbox_gt.py -o out-wave-equation-gt

python src/wave-equation/plot.py --ours_path out-wave-equation-ours --nn_path out-wave-equation-nn --gt_path out-wave-equation-gt -o out-wave-equation-plot.png --cmin -0.20 --cmax 0.20 --cmap RdBu_r
