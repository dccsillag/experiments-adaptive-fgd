#!/bin/sh -ex

for lr in 0.05 0.2; do
    for eps in 0.1 3; do
        out=out-teaser-ours/lr$lr-eps$eps
        mkdir -p $out && rm -f $out/*
        python src/teaser/sandbox_ours.py --output $out --lr $lr --eps $eps
    done
done
out=out-teaser-ours/lr0.05-eps0.1-fine
mkdir -p $out && rm -f $out/*
python src/teaser/sandbox_ours.py --output $out --lr 0.05 --eps 0.1 --n_epochs 200

mkdir -p out-teaser-nn && rm -f out-teaser-nn/*
python src/teaser/sandbox_nn.py

lr=0.05
for grid_size in 16 32 64 128 256 512; do
    out=out-teaser-naive-fgd/grid$grid_size-lr$lr
    mkdir -p $out && rm -f $out/*
    python src/teaser/sandbox_naive_fgd.py --output $out --grid_size $grid_size --lr $lr
done

mkdir -p out-teaser
mkdir -p out-teaser/nn
mkdir -p out-teaser/naive-fgd
mkdir -p out-teaser/ours

python src/teaser/plot.py --dir_ours out-teaser-ours/lr0.05-eps3/ --dir_naive_fgd out-teaser-naive-fgd/grid32-lr0.05/ --dir_nn out-teaser-nn/ --cmap viridis
python src/teaser/individual_plots.py --dir_ours out-teaser-ours/lr0.05-eps3/ --dir_naive_fgd out-teaser-naive-fgd/grid32-lr0.05/ --dir_nn out-teaser-nn/
python src/teaser/individual_plots_steps.py --dir_ours out-teaser-ours/lr0.05-eps3/ --dir_naive_fgd out-teaser-naive-fgd/grid32-lr0.05/ --dir_nn out-teaser-nn/
python src/teaser/plot_learning_curve.py --dir_ours out-teaser-ours/lr0.05-eps0.1-fine/ --dir_naive_fgd out-teaser-naive-fgd/grid%s-lr0.05/ --dir_nn out-teaser-nn/
