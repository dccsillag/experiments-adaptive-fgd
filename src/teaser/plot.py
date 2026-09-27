import datetime
import json
from argparse import ArgumentParser
from glob import glob

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.teaser.common import f_star

parser = ArgumentParser()
parser.add_argument("--dir_ours", required=True)
parser.add_argument("--dir_naive_fgd", required=True)
parser.add_argument("--dir_nn", required=True)
parser.add_argument("--cmap", required=True)
args = parser.parse_args()


def format_duration(delta: datetime.timedelta):
    if delta < datetime.timedelta(seconds=1):
        return f"{round(delta.microseconds / 1000)}ms"

    seconds = int(delta.total_seconds())
    periods = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("secs", 1)]

    parts = []
    for suffix, length in periods:
        if seconds >= length:
            value, seconds = divmod(seconds, length)
            parts.append(f"{value}{suffix}")

    return " ".join(parts) if parts else "0s"


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


DESIRED_STEPS = [10, 25, 50, 100]

fig, axs = plt.subplots(
    3,
    7,
    figsize=(10.4, 4.6),
    gridspec_kw={
        "width_ratios": [1.0] * 5 + [0.02] + [1.0],
    },
)
for ax in np.ravel(axs):
    ax.set_xticks([])
    ax.set_yticks([])

for i, step in enumerate(DESIRED_STEPS):
    axs[0, i].set_title(f"Step #{step}")

axs[0, 0].set_ylabel("Neural Network\n(w/ Fourier Embed.)")
for i, step in enumerate(DESIRED_STEPS):
    results = results_nn[step]
    axs[0, i].imshow(
        results,
        clim=(-0.5, +0.5),
        extent=[0, 1, 0, 1],
        cmap=args.cmap,
    )

axs[1, 0].set_ylabel("Naive Functional\nGrad. Descent")
for i, step in enumerate(DESIRED_STEPS):
    results = results_naive_fgd[step]
    axs[1, i].imshow(
        results,
        clim=(-0.5, +0.5),
        extent=[0, 1, 0, 1],
        cmap=args.cmap,
    )

axs[2, 0].set_ylabel(r"Our method")
for i, step in enumerate(DESIRED_STEPS):
    results = results_ours[step]
    axs[2, i].imshow(
        results,
        clim=(-0.5, +0.5),
        extent=[0, 1, 0, 1],
        cmap=args.cmap,
    )

for i in range(3):
    axs[i, -2].axis("off")

axs[0, -1].set_title(r"Target")
for i in range(3):
    xx, yy = jnp.meshgrid(jnp.linspace(0, 1, 200), jnp.linspace(0, 1, 200))

    axs[i, -1].imshow(
        jax.vmap(jax.vmap(f_star))(jnp.stack((xx, yy), axis=-1)),
        clim=(-0.5, +0.5),
        extent=[0, 1, 0, 1],
        cmap=args.cmap,
    )

fig.savefig("out-teaser.png", dpi=300)
plt.close()
