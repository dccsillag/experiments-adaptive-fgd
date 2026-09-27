import argparse
import json

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.functional_radiance.models.nerf_pytorch.utils.load_blender import (
    load_blender_data,
)

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", type=str)

args = parser.parse_args()


epoch_folder = Path(args.path)

str_epoch = epoch_folder.stem.split("_")[-1]

paths = sorted(list(epoch_folder.glob("*")))

vssim = []
vpsnr = []
vmse = []

with Path(f"{epoch_folder.parent}/args.txt").open("r") as f:
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
    gt_img = images[i_test[i]].astype(np.float32)

    mse = float(np.mean((prediction_img - gt_img) ** 2))
    psnr = float(peak_signal_noise_ratio(prediction_img, gt_img))
    ssim = float(
        structural_similarity(prediction_img, gt_img, channel_axis=-1, data_range=1.0)
    )

    vmse.append(mse)
    vpsnr.append(psnr)
    vssim.append(ssim)

print(
    json.dumps(
        {
            "epoch": int(str_epoch),
            "mse": float(np.mean(vmse)),
            "psnr": float(np.mean(vpsnr)),
            "ssim": float(np.mean(vssim)),
        }
    )
)
