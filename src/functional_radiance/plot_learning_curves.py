import datetime
import json
import os
from argparse import ArgumentParser
from glob import glob

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from tqdm import tqdm

from src.teaser.common import f_star

parser = ArgumentParser()
parser.add_argument("--results_dir", default="results/functional-radiance/")
args = parser.parse_args()

METRIC = "mse"
# METRIC = "ssim"
# METRIC = "psnr"


losses_fixed_grids = {}
for grid_size in [22, 32, 45, 64, 90]:
    with open(f"{args.results_dir}/ficus-fixed_grid_{grid_size}/{METRIC}.json") as file:
        losses_fixed_grids[grid_size] = json.load(file)["val"]

with open(
    "models/functional_radiance/nerf/ficus/small-160-log-nofine-batching-nrand6400-lr1e-4-niter15000/test-eval.jsonl"
) as file:
    losses_nn = [json.loads(line)[f"{METRIC}"] for line in file.readlines()]

with open(f"{args.results_dir}/ficus-ours/{METRIC}.json") as file:
    losses_ours = json.load(file)["val"]

# ----

plt.figure(figsize=(3.8, 4.8))

matplotlib.rc("text", usetex=True)

cmap = matplotlib.colormaps["Blues"]
for (grid_size, grid_results), t in zip(
    losses_fixed_grids.items(),
    jnp.linspace(0.25, 1, len(losses_fixed_grids), endpoint=False),
):
    plt.plot(
        jnp.arange(len(grid_results)),
        grid_results,
        label=f"Naive FGD ({grid_size**2} parameters)",
        color=cmap(t),
    )

plt.plot(
    jnp.arange(len(losses_ours)),
    losses_ours,
    label="Our method",
    color="green",
)

# 1 losses_nn <=> 24 steps with bs=80*80
#             <=> 1/4 epoch (24 images, 160*160)
plt.plot(
    jnp.arange(0, len(losses_nn)) / 4,
    losses_nn,
    label="Neural network",
    color="orange",
)

plt.yscale("log")
ax = plt.gca()
ax.yaxis.tick_right()
ax.set_yticks([4e-3, 5e-3, 6e-3, 7e-3, 8e-3])
ax.set_yticklabels(["4e-3", "5e-3", "6e-3", "7e-3", "8e-3"])
ax.minorticks_on()


plt.xlim(0, 70)
plt.ylim(3.1e-3, 9e-3)

plt.tight_layout()
plt.savefig("out-functional-radiance-plot.png", dpi=300)
plt.close()
