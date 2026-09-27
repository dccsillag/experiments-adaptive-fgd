from dataclasses import dataclass
from math import prod
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Antialiasing(Protocol):
    def get_hparams_json(self) -> dict:
        pass

    @property
    def support_radius(self) -> int:
        pass

    def kernel_function(self, uv1_delta: Float[Array, "2"]) -> Float[Array, ""]:
        pass

    def sample_for_rendering(
        self, *, shape: tuple[int, ...], rng
    ) -> tuple[
        Float[Array, "*shape 2"], Float[Array, "*shape"]
    ]:  # (perturbations, weights)
        pass


@dataclass
class BoxFilter(Antialiasing):
    subbox_scale: float

    def get_hparams_json(self) -> dict:
        return {
            "filter": "box",
            "subbox_scale": self.subbox_scale,
        }

    @property
    def support_radius(self) -> int:
        return 0

    def kernel_function(self, uv1_delta: Float[Array, "2"]) -> Float[Array, ""]:
        return (
            jnp.all(2 * jnp.abs(uv1_delta) <= self.subbox_scale) / self.subbox_scale**2
        )

    def sample_for_rendering(
        self, *, shape: tuple[int, ...], rng
    ) -> tuple[
        Float[Array, "*shape 2"], Float[Array, "*shape"]
    ]:  # (perturbations, weights)
        return (
            _weyl_sequence_2d(shape),
            jnp.ones(shape),
        )


@dataclass
class BsplineFilter(Antialiasing):
    importance_sample: bool

    def get_hparams_json(self) -> dict:
        return {
            "filter": "bspline",
            "importance_sample": self.importance_sample,
        }

    @property
    def support_radius(self) -> int:
        return 1

    def kernel_function(self, uv1_delta: Float[Array, "2"]) -> Float[Array, ""]:
        assert uv1_delta.shape == (2,)
        return jnp.prod(jax.vmap(self._bspline)(uv1_delta))

    def _bspline(self, t):
        return jnp.where(
            t <= -3 / 2,
            0.0,
            jnp.where(
                t <= -1 / 2,
                0.5 * (t + 3 / 2) ** 2,
                jnp.where(
                    t <= 1 / 2,
                    3 / 4 - t**2,
                    jnp.where(
                        t <= 3 / 2,
                        0.5 * (t - 3 / 2) ** 2,
                        0.0,
                    ),
                ),
            ),
        )

    def sample_for_rendering(
        self, *, shape: tuple[int, ...], rng
    ) -> tuple[
        Float[Array, "*shape 2"], Float[Array, "*shape"]
    ]:  # (perturbations, weights)
        if self.importance_sample:
            perturbations = jnp.sum(
                jax.random.uniform(
                    rng,
                    shape=(*shape, 2, 3),
                    minval=-0.5,
                    maxval=+0.5,
                ),
                axis=-1,
            )
            return (
                perturbations,
                jnp.ones(shape),
            )
        else:
            perturbations = jax.random.uniform(
                rng,
                shape=(*shape, 2),
                minval=-1.5,
                maxval=+1.5,
            )
            weights = 3**2 * jax.vmap(self.kernel_function)(
                perturbations.reshape((-1, 2))
            ).reshape(perturbations.shape[:-1])
            return (
                perturbations,
                weights,
            )


def _weyl_sequence_2d(shape: tuple[int, ...]) -> Float[Array, "*shape 2"]:
    """Generates a 2D Weyl sequence based on the generalized golden ratio."""

    return jax.random.uniform(
        jax.random.key(0),
        shape=shape + (2,),
        minval=-0.5,
        maxval=+0.5,
    )
