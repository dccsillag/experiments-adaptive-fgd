import argparse
from pathlib import Path

import numpy as np
from PIL import Image as im
from skimage.metrics import peak_signal_noise_ratio


def get_psnr(directory):
    dir = Path(directory)

    dataset = dir.parent.parent.parent.parent.parent.name
    fold = dir.parent.name

    print(f"{dataset=}")
    print(f"{fold=}")
    gt_dir = (
        Path("models/gt/")
        / dataset
        / Path("default/predictions")
        / fold
        / Path("e000000")
    )
    ground_truth_images = sorted(gt_dir.glob("*"))
    assert len(ground_truth_images) > 0, (
        "couldn't find any ground truth images. Have you run the gt ('ground truth') prediction script (`python src/models/gt/predict.py`)?"
    )

    v_psnr = []
    for img in ground_truth_images:
        model_img = np.array(im.open(dir / img.name))
        gt_img = np.array(im.open(img))
        v_psnr.append(peak_signal_noise_ratio(gt_img, model_img))

    return float(np.mean(v_psnr))


def get_nerf_prediction_folder(hparams, fold="val"):
    is_default = (
        hparams["size"] == 0.025
        and hparams["n_train"] == 12
        and hparams["test_skip"] == 8
        and hparams["val_skip"] == 8
        and hparams["dataset"] == "lego_small"
    )

    if not is_default or fold != "val":
        return None

    return "./models/nerf_pytorch/lego_small/default/blender_paper_lego/predictions/val/e029999"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--directory",
        default="./models/nerf_pytorch/lego_small/default/blender_paper_lego/predictions/val/e029999",
    )

    args = parser.parse_args()

    print(get_psnr(args.directory))
