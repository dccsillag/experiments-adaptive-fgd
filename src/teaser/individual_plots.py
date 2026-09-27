import datetime
import json
from argparse import ArgumentParser
from glob import glob

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from tqdm import tqdm

from src.teaser.common import f_star

parser = ArgumentParser()
parser.add_argument("--dir_ours", required=True)
parser.add_argument("--dir_naive_fgd", required=True)
parser.add_argument("--dir_nn", required=True)
args = parser.parse_args()


results_ours = [np.load(fname) for fname in sorted(glob(f"{args.dir_ours}/*.npy"))]
with open(f"{args.dir_ours}/timings.json") as file:
    timings_ours = json.load(file)

results_naive_fgd = [
    np.load(fname) for fname in sorted(glob(f"{args.dir_naive_fgd}/*.npy"))
]
with open(f"{args.dir_naive_fgd}/timings.json") as file:
    timings_naive_fgd = json.load(file)

results_nn = [np.load(fname) for fname in sorted(glob(f"{args.dir_nn}/*.npy"))]
with open(f"{args.dir_nn}/timings.json") as file:
    timings_nn = json.load(file)


def get_result(results, timings, desired_time):
    i = np.argmin(np.abs(np.array(timings) - desired_time))
    return results[i]


def mkimage(results: Float[Array, "w h"], path: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        results,
        clim=(-0.5, +0.5),
        extent=(0, 1, 0, 1),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path)
    plt.close()


for n_millis in tqdm([100, 200, 300, 400, 500, 600, 700, 800, 900, 250, 750]):
    time = n_millis / 1000
    mkimage(
        get_result(results_nn, timings_nn, time),
        f"out-teaser/nn/result-{n_millis}ms.png",
    )
    mkimage(
        get_result(results_ours, timings_ours, time),
        f"out-teaser/ours/result-{n_millis}ms.png",
    )
    mkimage(
        get_result(results_naive_fgd, timings_naive_fgd, time),
        f"out-teaser/naive-fgd/result-{n_millis}ms.png",
    )

for n_secs in tqdm([1, 2, 3, 4, 5, 10, 30]):
    time = n_secs
    mkimage(
        get_result(results_nn, timings_nn, time), f"out-teaser/nn/result-{n_secs}s.png"
    )
    mkimage(
        get_result(results_ours, timings_ours, time),
        f"out-teaser/ours/result-{n_secs}s.png",
    )
    mkimage(
        get_result(results_naive_fgd, timings_naive_fgd, time),
        f"out-teaser/naive-fgd/result-{n_secs}s.png",
    )

for n_mins in tqdm([1, 2, 3, 4, 5, 10]):
    time = n_secs
    mkimage(
        get_result(results_nn, timings_nn, time), f"out-teaser/nn/result-{n_mins}m.png"
    )
    mkimage(
        get_result(results_ours, timings_ours, time),
        f"out-teaser/ours/result-{n_mins}m.png",
    )
    mkimage(
        get_result(results_naive_fgd, timings_naive_fgd, time),
        f"out-teaser/naive-fgd/result-{n_mins}m.png",
    )

xx, yy = jnp.meshgrid(jnp.linspace(0, 1, 200), jnp.linspace(0, 1, 200))

mkimage(
    jax.vmap(jax.vmap(f_star))(jnp.stack((xx, yy), axis=-1)), "out-teaser/target.png"
)
