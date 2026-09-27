import json
from pathlib import Path
from typing import Literal

import cv2
import jax.numpy as jnp
import numpy as np
from imageio import imread
from jaxtyping import Array, Float
from skimage.transform import resize

from src.functional_radiance.utils.models import Camera, CameraAndImage, Dataset


def rgba2rgb(
    rgba: Float[Array, "h w 4"], *, background: Float[Array, "3"]
) -> Float[Array, "h w 3"]:
    # Blends with a black background. Both input and output should have channel values in [0, 1].
    # Adapted from https://stackoverflow.com/a/58748986.

    row, col, ch = rgba.shape
    assert ch == 4, "RGBA image has 4 channels."

    rgb = jnp.zeros((row, col, 3))
    a = rgba[:, :, 3]
    rgb = rgb.at[:, :, 0].set(rgba[:, :, 0] * a + background[0] * (1 - a))
    rgb = rgb.at[:, :, 1].set(rgba[:, :, 1] * a + background[1] * (1 - a))
    rgb = rgb.at[:, :, 2].set(rgba[:, :, 2] * a + background[2] * (1 - a))

    return rgb


def load_nerfsynthetic_data(
    case: str,
    split: Literal["train"] | Literal["val"] | Literal["test"],
    *,
    background: Float[Array, "3"],
    normalized_parametrization: bool = True,
    rays_at_centers: bool = False,
    n: int | None,
    skip: int | None,
) -> Dataset:
    directory = Path(f"data/nerf_synthetic/{case}")

    with open(directory / f"transforms_{split}.json") as file:
        transforms = json.load(file)

    images = []
    poses = []
    for frame in transforms["frames"]:
        image = (
            jnp.asarray(imread(directory / f"{frame['file_path']}.png")).astype(
                jnp.float32
            )
            / 255
        )
        if image.shape[-1] == 4:
            image = rgba2rgb(image, background=background)
        assert image.ndim == 3

        pose = jnp.array(frame["transform_matrix"])

        images.append(image)
        poses.append(pose)

    # taken from nerf
    H, W = images[0].shape[:2]
    camera_angle_x = float(transforms["camera_angle_x"])
    focal = float(0.5 * W / jnp.tan(0.5 * camera_angle_x))

    return Dataset.from_list(
        [
            CameraAndImage(
                camera=Camera(
                    pose=jnp.array(pose),
                    pose_det=jnp.linalg.det(jnp.array(pose)[:3, :3]),
                    focal=focal,
                    normalized_parametrization=normalized_parametrization,
                    rays_at_centers=rays_at_centers,
                ),
                pixels=jnp.array(image),
            )
            for image, pose in zip(images, poses)
        ][:n][::skip],
        normalized_parametrization=normalized_parametrization,
        rays_at_centers=rays_at_centers,
    )


def resize_images(data: Dataset, *, factor: float) -> Dataset:
    return Dataset.from_list(
        [
            CameraAndImage(
                camera=Camera(
                    pose=data[i].camera.pose,
                    pose_det=data[i].camera.pose_det,
                    focal=data[i].camera.focal * factor,
                    normalized_parametrization=data[
                        i
                    ].camera.normalized_parametrization,
                    rays_at_centers=data[i].camera.rays_at_centers,
                ),  # type: ignore
                pixels=jnp.array(
                    cv2.resize(
                        np.array(data[i].pixels),
                        (
                            round(data[i].pixels.shape[0] * factor),
                            round(data[i].pixels.shape[1] * factor),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                ),
            )
            for i in range(len(data))
        ],
        normalized_parametrization=data.camera_normalized_parametrization,
        rays_at_centers=data.rays_at_centers,
    )
