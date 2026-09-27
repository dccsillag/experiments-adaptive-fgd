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
parser.add_argument("--results_dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--loss", required=True, choices=["mse", "bce"])
args = parser.parse_args()

losses_fixed_grids = {}
for depth in [2, 4, 8, 12]:
    with open(f"{args.results_dir}/fgd_naive_{depth}.json") as file:
        losses_fixed_grids[depth] = json.load(file)["learning_curve"]["test_loss"]

with open(f"{args.results_dir}/nn.json") as file:
    losses_nn = json.load(file)["learning_curve"]["test_loss"]

with open(f"{args.results_dir}/ours.json") as file:
    losses_ours = json.load(file)["learning_curve"]["test_loss"]

# ----

plt.figure(figsize=(2.2, 2.4))

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

plt.plot(
    jnp.arange(0, len(losses_nn)),
    losses_nn,
    label="Neural network",
    color="orange",
)

match args.loss:
    case "mse":
        plt.ylabel("Test loss (MSE)")
    case "bce":
        plt.ylabel("Test loss (cross-entropy)")
plt.xlabel("Optimization steps")

plt.yscale("log")

plt.gca().yaxis.tick_right()

match args.loss:
    case "mse":
        plt.xlim(0, 200)
        plt.ylim(3.4e-2, 0.5)
    case "bce":
        plt.xlim(0, 60)
        plt.ylim(2.2e-1, 1.5e0)

plt.savefig(args.output, dpi=300, bbox_inches="tight")
plt.close()
