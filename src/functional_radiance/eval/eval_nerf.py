import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from lpips import LPIPS
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from src.functional_radiance.models.nerf_pytorch.utils.load_blender import (
    load_blender_data,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    type=str,
)

args = parser.parse_args()

lpips_impl = LPIPS(net="vgg")
lpips_fn = lambda a, b: float(
    np.squeeze(
        lpips_impl(
            torch.permute(
                torch.from_numpy(np.asarray(2 * a.astype(np.float32) - 1)), (2, 0, 1)
            )[None, ...],
            torch.permute(
                torch.from_numpy(np.asarray(2 * b.astype(np.float32) - 1)), (2, 0, 1)
            )[None, ...],
        )
        .detach()
        .numpy()
    )
)


folder = args.path

testset_list = sorted(list(Path(folder).glob("testset_*")))
last_epoch_folder = testset_list[-1]
print(f"{last_epoch_folder=}")
paths = sorted(list(last_epoch_folder.glob("*")))

vlpips = []
vssim = []
vpsnr = []

with Path(f"{folder}/args.txt").open("r") as f:
    args_lines = f.readlines()

hparams = {}

for line in args_lines:
    key = line.split()[0]
    val = line.split()[2]

    if val.isdigit():
        val = int(val)
    else:
        try:
            val = float(val)
        except ValueError:
            pass

    hparams[key] = val

images, poses, render_poses, hwf, i_split = load_blender_data(
    hparams["datadir"], hparams["size"], hparams["testskip"]
)

if hparams["white_bkgd"]:
    images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
else:
    images = images[..., :3]

i_train, i_val, i_test = i_split

for i, path in enumerate(paths):
    prediction_img = (np.asarray(Image.open(path)) / 255.0).astype(np.float32)
    gt_img = images[i_test[i]]
    psnr = float(peak_signal_noise_ratio(prediction_img, gt_img))
    ssim = float(
        structural_similarity(prediction_img, gt_img, channel_axis=-1, data_range=1.0)
    )
    lpips = float(lpips_fn(prediction_img, gt_img))

    vpsnr.append(psnr)
    vssim.append(ssim)
    vlpips.append(lpips)

with (Path(folder) / "psnr.txt").open("w") as f:
    f.write(str(float(np.mean(vpsnr))))

with (Path(folder) / "ssim.txt").open("w") as f:
    f.write(str(float(np.mean(vssim))))

with (Path(folder) / "lpips.txt").open("w") as f:
    f.write(str(float(np.mean(vlpips))))
