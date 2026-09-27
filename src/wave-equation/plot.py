import datetime
import json
from argparse import ArgumentParser

import humanize
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("--ours_path", required=True)
parser.add_argument("--nn_path", required=True)
parser.add_argument("--gt_path", required=True)
parser.add_argument("--cmin", type=float, required=True)
parser.add_argument("--cmax", type=float, required=True)
parser.add_argument("--cmap", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()


def load(path) -> list[tuple[int, np.ndarray]]:
    frames = np.load(f"{path}/results.npy")
    with open(f"{path}/data.json") as file:
        data = json.load(file)
    return frames, data


def format_duration(delta: datetime.timedelta):
    seconds = int(delta.total_seconds())
    periods = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("secs", 1)]

    parts = []
    for suffix, length in periods:
        if seconds >= length:
            value, seconds = divmod(seconds, length)
            parts.append(f"{value}{suffix}")

    return " ".join(parts) if parts else "0s"


ours_frames, ours_data = load(args.ours_path)
nn_frames, nn_data = load(args.nn_path)
gt_frames, gt_data = load(args.gt_path)

assert ours_frames.shape[0] == nn_frames.shape[0]
assert ours_frames.shape[0] == gt_frames.shape[0]
n_frames = nn_frames.shape[0]

fig, axs = plt.subplots(3, n_frames, figsize=(10.4, 5.2))
for ax in np.ravel(axs):
    ax.set_xticks([])
    ax.set_yticks([])

ours_duration = format_duration(datetime.timedelta(seconds=ours_data["training time"]))
axs[1, 0].set_ylabel("Ours\n$\\bf{(" + ours_duration + ")}$")
for i in range(n_frames):
    axs[1, i].imshow(ours_frames[i, ...], clim=(args.cmin, args.cmax), cmap=args.cmap)

nn_duration = format_duration(datetime.timedelta(seconds=nn_data["training time"]))
axs[0, 0].set_ylabel("Neural Network (PINN)\n$\\bf{(" + nn_duration + ")}$")
for i in range(n_frames):
    axs[0, i].imshow(nn_frames[i, ...], clim=(args.cmin, args.cmax), cmap=args.cmap)
    axs[0, i].set_title(f"$t = {nn_data['ts'][i]}$")

gt_duration = format_duration(datetime.timedelta(seconds=gt_data["training time"]))
axs[2, 0].set_ylabel("Reference Solution")
for i in range(n_frames):
    axs[2, i].imshow(gt_frames[i, ...], clim=(args.cmin, args.cmax), cmap=args.cmap)

plt.tight_layout()
plt.savefig(args.output, dpi=300)
plt.close()
