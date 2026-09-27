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

results_naive_fgd = [
    np.load(fname) for fname in sorted(glob(f"{args.dir_naive_fgd}/*.npy"))
]

results_nn = [np.load(fname) for fname in sorted(glob(f"{args.dir_nn}/*.npy"))]


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


for step in tqdm(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        150,
        200,
    ]
):
    mkimage(
        results_nn[step],
        f"out-teaser/nn/result-step{step:04}.png",
    )
    mkimage(
        results_ours[step],
        f"out-teaser/ours/result-step{step:04}.png",
    )
    mkimage(
        results_naive_fgd[step],
        f"out-teaser/naive-fgd/result-step{step:04}.png",
    )

xx, yy = jnp.meshgrid(jnp.linspace(0, 1, 1000), jnp.linspace(0, 1, 1000))

mkimage(
    jax.vmap(jax.vmap(f_star))(jnp.stack((xx, yy), axis=-1)), "out-teaser/target.png"
)
