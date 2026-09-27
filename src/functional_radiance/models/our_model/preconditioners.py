from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from icecream import ic  # type: ignore
from jaxtyping import Array, Float, Int, PyTree

from src.functional_radiance.utils.models import CameraAndImage, Dataset


class Preconditioner(Protocol):
    # corresponds to diagonal preconditioning, i.e., changing the measure over the L^2 spcae.

    def get_hparams_json(self) -> dict:
        pass

    def __call__(self, x: Float[Array, "3"], *, full_data: Dataset) -> Float[Array, ""]:
        pass


@dataclass
class ConstantPreconditioner(Preconditioner):
    r"""Constant preconditioning, i.e., no preconditioning at all."""

    def get_hparams_json(self) -> dict:
        return {
            "preconditioner": "constant",
        }

    def __call__(self, x: Float[Array, "3"], *, full_data: Dataset) -> Float[Array, ""]:
        return 1.0  # type: ignore


@dataclass
class SupervisionPreconditioner(Preconditioner):
    r"""Supervision-based preconditioning, i.e., preconditioning at x is given by the proporition of training cameras that view the point x."""

    t_near: float
    t_far: float
    pow: float

    def get_hparams_json(self) -> dict:
        return {
            "preconditioner": "supervision",
            "t_near": self.t_near,
            "t_far": self.t_far,
            "pow": self.pow,
        }

    def __call__(self, x: Float[Array, "3"], *, full_data: Dataset) -> Float[Array, ""]:
        def check_if_in_frustrum(datum: CameraAndImage):
            uv = datum.camera.world_position_to_uv(x)
            ij = datum.uv_to_ij(uv)

            ray = datum.camera.ray_at_normalized(uv)
            t = ((x - ray.origin) @ ray.direction) / (ray.direction @ ray.direction)

            return (
                datum.is_in_bounds(ij)
                & (self.t_near - 1e-4 <= t)
                & (t <= self.t_far + 1e-4)
            )

        return (
            jnp.mean(full_data.map(check_if_in_frustrum, batch_size=None)) ** self.pow
        )


@dataclass
class ForegroundPreconditioner(Preconditioner):
    t_near: float
    t_far: float
    background_lr_mult: float

    def get_hparams_json(self) -> dict:
        return {
            "preconditioner": "foreground",
            "t_near": self.t_near,
            "t_far": self.t_far,
            "background_lr_mult": self.background_lr_mult,
        }

    def __call__(self, x: Float[Array, "3"], *, full_data: Dataset) -> Float[Array, ""]:
        def check_if_background(datum: CameraAndImage):
            uv = datum.camera.world_position_to_uv(x)
            ij = datum.uv_to_ij(uv)

            return jnp.where(
                datum.is_in_bounds(ij),
                jnp.all(datum.pixels[*ij] == 1.0),
                True,
            )

        return jnp.where(
            jnp.any(full_data.map(check_if_background, batch_size=None)),
            self.background_lr_mult,
            1.0,
        )


@dataclass
class WeightPreconditioner(Preconditioner):
    t_near: float
    t_far: float

    def get_hparams_json(self) -> dict:
        return {
            "preconditioner": "supervision",
            "t_near": self.t_near,
            "t_far": self.t_far,
        }

    def __call__(self, x: Float[Array, "3"], *, full_data: Dataset) -> Float[Array, ""]:
        def get_weight(datum: CameraAndImage):
            uv = datum.camera.world_position_to_uv(x)
            ij = datum.uv_to_ij(uv)

            ray = datum.camera.ray_at_normalized(uv)
            t = ((x - ray.origin) @ ray.direction) / (ray.direction @ ray.direction)

            return (
                datum.is_in_bounds(ij)
                & (self.t_near - 1e-4 <= t)
                & (t <= self.t_far + 1e-4)
            ) / t**2

        return jnp.sum(full_data.map(get_weight, batch_size=None)) / (
            len(full_data) / self.t_near**2
        )


@dataclass
class FarFromCameraPreconditioner(Preconditioner):
    t_near: float
    t_far: float
    dist_pow: float

    def get_hparams_json(self) -> dict:
        return {
            "preconditioner": "supervision",
            "t_near": self.t_near,
            "t_far": self.t_far,
            "dist_pow": self.dist_pow,
        }

    def __call__(self, x: Float[Array, "3"], *, full_data: Dataset) -> Float[Array, ""]:
        def get_weight(datum: CameraAndImage):
            uv = datum.camera.world_position_to_uv(x)
            ij = datum.uv_to_ij(uv)

            ray = datum.camera.ray_at_normalized(uv)
            t = ((x - ray.origin) @ ray.direction) / (ray.direction @ ray.direction)
            t01 = (t - self.t_near) / (self.t_far - self.t_near)

            return (
                datum.is_in_bounds(ij)
                & (self.t_near - 1e-4 <= t)
                & (t <= self.t_far + 1e-4)
            ) * (t01**self.dist_pow)

        return jnp.mean(full_data.map(get_weight, batch_size=None))
