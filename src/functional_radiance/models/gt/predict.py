import argparse

import aim
import jax.numpy as jnp

run = aim.Run(experiment="ground_truth")

from src.functional_radiance.utils.data import load_nerfsynthetic_data, resize_images

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "-s",
    "--size",
    default=0.025,
    help="relative image size in (0.0, 1.0]",
)

parser.add_argument(
    "-n",
    "--n_train",
    default=12,
    help="number of train images (obs: possibly weird distribution, since there's no shuffle)",
)

parser.add_argument(
    "--test_skip",
    default=8,
)

parser.add_argument(
    "--val_skip",
    default=8,
)

parser.add_argument(
    "-d",
    "--dataset",
    default="lego",
    choices=["lego"],
    help="image dataset",
)

args = parser.parse_args()

hparams = vars(args)

run["hparams"] = hparams

import os

import numpy as np
from jax import config

config.update("jax_enable_x64", True)


from PIL import Image as im

BACKGROUND = jnp.array((1.0, 1.0, 1.0))
RESIZE_FACTOR = hparams["size"]

N_TRAIN = hparams["n_train"]


dataset = hparams["dataset"]

if dataset == "lego":
    # Load data:
    train_data = resize_images(
        load_nerfsynthetic_data("lego", "train", background=BACKGROUND),
        factor=RESIZE_FACTOR,
    )[:N_TRAIN]

    val_data = resize_images(
        load_nerfsynthetic_data("lego", "val", background=BACKGROUND),
        factor=RESIZE_FACTOR,
    )[:: hparams["val_skip"]]

    test_data = resize_images(
        load_nerfsynthetic_data("lego", "test", background=BACKGROUND),
        factor=RESIZE_FACTOR,
    )[:: hparams["test_skip"]]

    print(f"Image sizes: {train_data[0].pixels.shape = }")

# Training loop, together with eval at each step:

output_folder = "models/gt/lego_small/default"

# Setup where we will save the outputs:
os.makedirs(output_folder)

os.makedirs(output_folder + f"/predictions/train/e{0:06}")
os.makedirs(output_folder + f"/predictions/val/e{0:06}")
os.makedirs(output_folder + f"/predictions/test/e{0:06}")

train_psnrs = []
val_psnrs = []

aim_images = []
for i, train_datum in enumerate(train_data):
    train_img = train_datum.pixels

    train_img = np.array(train_img)
    out_img = im.fromarray(np.array(255.0 * np.array(train_img), dtype=np.uint8))
    output_path = f"{output_folder}/predictions/train/e{0:06}/{i:06}.png"
    out_img.save(output_path)

    aim_images.append(aim.Image(output_path))

run.track(aim_images, name="render", step=0, epoch=0, context={"subset": "train"})

aim_images = []
for i, val_datum in enumerate(val_data):
    val_img = val_datum.pixels

    val_img = np.array(val_img)
    out_img = im.fromarray(np.array(255.0 * np.array(val_img), dtype=np.uint8))
    output_path = f"{output_folder}/predictions/val/e{0:06}/{i:06}.png"
    out_img.save(output_path)
    aim_images.append(aim.Image(output_path))

run.track(aim_images, name="render", step=0, epoch=0, context={"subset": "validation"})

aim_images = []
for i, test_datum in enumerate(test_data):
    test_img = test_datum.pixels

    test_img = np.array(test_img)
    out_img = im.fromarray(np.array(255.0 * np.array(test_img), dtype=np.uint8))
    output_path = f"{output_folder}/predictions/test/e{0:06}/{i:06}.png"
    out_img.save(output_path)
    aim_images.append(aim.Image(output_path))

run.track(aim_images, name="render", step=0, epoch=0, context={"subset": "test"})
