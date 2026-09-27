from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.functional_radiance.utils.sh_utils import eval_sh


# An RKHS over S^2.
class RKHS(Protocol):
    def kernel(self, x: Float[Array, "3"], y: Float[Array, "3"]) -> Float[Array, ""]:
        pass


@dataclass
class ConstantRKHS(RKHS):
    def kernel(self, x: Float[Array, "3"], y: Float[Array, "3"]) -> Float[Array, ""]:
        return 1.0  # type: ignore


@dataclass
class ZonalRKHS(RKHS):
    bandwidth: float

    def kernel(self, x: Float[Array, "3"], y: Float[Array, "3"]) -> Float[Array, ""]:
        p1 = x / jnp.linalg.norm(x)
        p2 = y / jnp.linalg.norm(y)
        return jnp.exp(-2 * self.bandwidth * (1 - p1 @ p2))


@dataclass
class SphericalHarmonicsRKHS(RKHS):
    deg: int

    def kernel(self, x: Float[Array, "3"], y: Float[Array, "3"]) -> Float[Array, ""]:
        p1 = x / jnp.linalg.norm(x)
        p2 = y / jnp.linalg.norm(y)

        d = (self.deg + 1) ** 2
        return sum(
            [
                eval_sh(self.deg, jnp.zeros((1, d)).at[0, i].set(1.0), p1)[0]
                * eval_sh(self.deg, jnp.zeros((1, d)).at[0, i].set(1.0), p2)[0]
                for i in range(d)
            ]
        )
