import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Self

import dill  # type: ignore
import jax
import jax.numpy as jnp
from icecream import ic  # type: ignore
from jax.experimental import checkify
from jaxtyping import Array, Float, Int, PyTree
from tqdm import tqdm  # type: ignore

from src.functional_radiance.utils.general import (
    BaseLearner,
    JaxRng,
    cumulative_trapezoid,
    jax_predict,
    viewing_direction_to_unit_circle,
)
from src.functional_radiance.utils.models import Camera, CameraAndImage, Ray


class DensityRegularization(Protocol):
    def grad_over_ray(
        self,
        *,
        t: Float[Array, ""],
        rendered_full: Float[Array, "3"],
        rendered_up_to_t: Float[Array, "3"],
        transmittance_at_t: Float[Array, ""],
        density_at_t: Float[Array, ""],
        color_at_t: Float[Array, "3"],
    ) -> Float[Array, ""]:
        pass

    def grad_over_space(
        self,
        x: Float[Array, "d"],
        *,
        density_function: Callable[[Float[Array, "d"]], Float[Array, ""]],
    ) -> Float[Array, ""]:
        pass


@dataclass
class PushdownRegularization(DensityRegularization):
    r"""Regularization by $\lambda \int \sigma(x) dx$"""

    lambda_: float
    threshold: float

    def grad_over_ray(
        self,
        *,
        t: Float[Array, ""],
        rendered_full: Float[Array, "3"],
        rendered_up_to_t: Float[Array, "3"],
        transmittance_at_t: Float[Array, ""],
        density_at_t: Float[Array, ""],
        color_at_t: Float[Array, "3"],
    ) -> Float[Array, ""]:
        return 0.0  # type: ignore

    def grad_over_space(
        self,
        x: Float[Array, "d"],
        *,
        density_function: Callable[[Float[Array, "d"]], Float[Array, ""]],
    ) -> Float[Array, ""]:
        density_at_x = density_function(x)
        return self.lambda_ * (density_at_x <= self.threshold)


@dataclass
class LpRegularization(DensityRegularization):
    r"""Regularizes by $lambda \int |\sigma(x)|^p dx$."""

    p: float
    lambda_: float

    def grad_over_ray(
        self,
        *,
        t: Float[Array, ""],
        rendered_full: Float[Array, "3"],
        rendered_up_to_t: Float[Array, "3"],
        transmittance_at_t: Float[Array, ""],
        density_at_t: Float[Array, ""],
        color_at_t: Float[Array, "3"],
    ) -> Float[Array, ""]:
        return 0.0  # type: ignore

    def grad_over_space(
        self,
        x: Float[Array, "d"],
        *,
        density_function: Callable[[Float[Array, "d"]], Float[Array, ""]],
    ) -> Float[Array, ""]:
        density_at_x = density_function(x)
        return jnp.where(
            jnp.abs(density_at_x) <= 1e-6,  # i.e., it's basically zero
            0.0,
            self.lambda_
            * self.p
            * jnp.abs(density_at_x)
            ** (self.p - 1)  # this would explode for density_at_x=0 and p<1
            * jnp.sign(density_at_x),
        )


@dataclass
class OccupancyRegularization(DensityRegularization):
    r"""Regularizes by $lambda \int relu(\sigma(x)) dx$."""

    lambda_: float

    def grad_over_ray(
        self,
        *,
        t: Float[Array, ""],
        rendered_full: Float[Array, "3"],
        rendered_up_to_t: Float[Array, "3"],
        transmittance_at_t: Float[Array, ""],
        density_at_t: Float[Array, ""],
        color_at_t: Float[Array, "3"],
    ) -> Float[Array, ""]:
        return 0.0  # type: ignore

    def grad_over_space(
        self,
        x: Float[Array, "d"],
        *,
        density_function: Callable[[Float[Array, "d"]], Float[Array, ""]],
    ) -> Float[Array, ""]:
        density_at_x = density_function(x)
        return jnp.where(density_at_x > 0, self.lambda_, 0.0)


@dataclass
class LpRegularizationAlongRay(DensityRegularization):
    r"""Regularizes by $lambda \E_I[ 1/P \sum_j \int_{t_\near}^{t_\far} |\sigma(\gamma_j(t))|^p dt ]$."""

    p: float
    lambda_: float

    def grad_over_ray(
        self,
        *,
        t: Float[Array, ""],
        rendered_full: Float[Array, "3"],
        rendered_up_to_t: Float[Array, "3"],
        transmittance_at_t: Float[Array, ""],
        density_at_t: Float[Array, ""],
        color_at_t: Float[Array, "3"],
    ) -> Float[Array, ""]:
        return jnp.where(
            jnp.abs(density_at_t) <= 1e-6,  # i.e., it's basically zero
            0.0,
            self.lambda_
            * self.p
            * jnp.abs(density_at_t)
            ** (self.p - 1)  # this would explode for density_at_t=0 and p<1
            * jnp.sign(density_at_t),
        )

    def grad_over_space(
        self,
        x: Float[Array, "d"],
        *,
        density_function: Callable[[Float[Array, "d"]], Float[Array, ""]],
    ) -> Float[Array, ""]:
        return 0.0  # type: ignore


@dataclass
class PlenoctreeSparsityRegularization(DensityRegularization):
    r"""Regularizes by $c \int |1 - \exp(-\lambda \sigma(x))| dx$."""

    lambda_: float
    c: float

    def grad_over_ray(
        self,
        *,
        t: Float[Array, ""],
        rendered_full: Float[Array, "3"],
        rendered_up_to_t: Float[Array, "3"],
        transmittance_at_t: Float[Array, ""],
        density_at_t: Float[Array, ""],
        color_at_t: Float[Array, "3"],
    ) -> Float[Array, ""]:
        return 0.0  # type: ignore

    def grad_over_space(
        self,
        x: Float[Array, "d"],
        *,
        density_function: Callable[[Float[Array, "d"]], Float[Array, ""]],
    ) -> Float[Array, ""]:
        density_at_x = density_function(x)
        return (
            self.c
            * jnp.sign(density_at_x)
            * self.lambda_
            * jnp.exp(-self.lambda_ * density_at_x)
        )
