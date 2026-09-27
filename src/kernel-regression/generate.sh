#!/bin/sh -ex

mkdir -p data
[ -f data/cod-rna.dat ] || curl -L https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna -o data/cod-rna.dat

for loss in mse bce
do
    python src/kernel-regression/nn.py --results-file results/kernel-$loss/nn.json --loss $loss

    python src/kernel-regression/ours.py --results-file results/kernel-$loss/ours.json --loss $loss

    for depth in 2 4 8 12 16
    do
        python src/kernel-regression/fgd_naive.py --depth $depth --results-file results/kernel-$loss/fgd_naive_$depth.json --loss $loss
    done

    python src/kernel-regression/plot_learning_curve.py --results_dir results/kernel-$loss/ --output out-kernel-$loss-plot.png --loss $loss
done
