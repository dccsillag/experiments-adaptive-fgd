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
parser.add_argument("--dir_ours", required=True)
parser.add_argument("--dir_naive_fgd", required=True)
parser.add_argument("--dir_nn", required=True)
args = parser.parse_args()


results_ours = [
    np.load(fname) for fname in tqdm(sorted(glob(f"{args.dir_ours}/*.npy")))
]
with open(f"{args.dir_ours}/timings.json") as file:
    timings_ours = json.load(file)

naive_fgd_grid_sizes = sorted(
    [
        int(x.split("-")[0].removeprefix("grid"))
        for x in tqdm(os.listdir("out-teaser-naive-fgd"))
    ]
)
results_naive_fgds = {}
timings_naive_fgds = {}
for grid_size in naive_fgd_grid_sizes:
    results_naive_fgds[grid_size] = [
        np.load(fname)
        for fname in tqdm(sorted(glob(f"{args.dir_naive_fgd}/*.npy" % grid_size)))
    ]
    with open(f"{args.dir_naive_fgd}/timings.json" % grid_size) as file:
        timings_naive_fgds[grid_size] = json.load(file)

results_nn = [np.load(fname) for fname in tqdm(sorted(glob(f"{args.dir_nn}/*.npy")))]
with open(f"{args.dir_nn}/timings.json") as file:
    timings_nn = json.load(file)


def get_result(results, timings, desired_time):
    i = np.argmin(np.abs(np.array(timings) - desired_time))
    return results[i]


xx, yy = jnp.meshgrid(jnp.linspace(0, 1, 1024), jnp.linspace(0, 1, 1024))
ground_truth = jax.vmap(jax.vmap(f_star))(jnp.stack((xx, yy), axis=-1))


LOSS_FLOOR = 2e-5


def get_loss(result):
    return 0.5 * jnp.mean((result[4::, 4::] - ground_truth[4::, 4::]) ** 2) + LOSS_FLOOR


plt.figure(figsize=(3.8, 4.8))

matplotlib.rc("text", usetex=True)

cmap = matplotlib.colormaps["Blues"]
for grid_size, t in zip(
    naive_fgd_grid_sizes,
    jnp.linspace(0.25, 1, len(naive_fgd_grid_sizes), endpoint=False),
):
    plt.plot(
        [get_loss(result) for result in tqdm(results_naive_fgds[grid_size])],
        label=f"Naive FGD ({grid_size**2} parameters)",
        color=cmap(t),
    )

plt.plot(
    [get_loss(result) for result in tqdm(results_ours)],
    label="Our method",
    color="green",
)

plt.plot(
    [get_loss(result) for result in results_nn],
    label="Neural network",
    color="orange",
)

plt.axhline(y=LOSS_FLOOR, color="k", alpha=0.3, linestyle="--")

plt.ylabel("Distance to solution")
plt.xlabel("Optimization steps")

plt.yscale("log")

plt.gca().yaxis.tick_right()

plt.xlim(0, 200)
plt.ylim(1e-5, 6e-2)

plt.savefig("out-teaser-plot.png", dpi=300)
plt.close()
