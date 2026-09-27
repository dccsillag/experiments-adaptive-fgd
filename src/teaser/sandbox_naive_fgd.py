import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from imageio import imread
from jaxtyping import Array, Float
from tqdm import tqdm

from src.teaser.common import f_star

parser = ArgumentParser()
parser.add_argument("--grid_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--n_epochs", type=int, default=2_000)
parser.add_argument("--output", required=True)
args = parser.parse_args()


BATCH_SIZE = 1024


@jdc.pytree_dataclass
class Representation:
    grid: Float[Array, "w h"]

    @jax.jit
    def __call__(self, x):
        w, h = self.grid.shape
        i = jnp.floor(x[0] * w).astype(jnp.int32)
        j = jnp.floor(x[1] * h).astype(jnp.int32)
        i = jnp.clip(i, 0, w - 1)
        j = jnp.clip(j, 0, h - 1)
        return self.grid[i, j]

    @classmethod
    def zeros(cls, *, grid_size):
        return Representation(grid=jnp.zeros((grid_size, grid_size)))

    @jax.jit
    def __add__(self, other: "Representation"):
        return Representation(self.grid + other.grid)

    @jax.jit
    def __sub__(self, other: "Representation"):
        return Representation(self.grid - other.grid)

    @jax.jit
    def __mul__(self, other: float):
        return Representation(other * self.grid)

    @jax.jit
    def __rmul__(self, other: float):
        return Representation(self.grid * other)

    @partial(jax.jit, static_argnames=("f",))
    def approx(
        self,
        f: Callable[["Representation", Float[Array, "2"]], Float[Array, ""]],
        *,
        rng,
    ) -> tuple["Representation", float]:
        h, w = self.grid.shape

        xs, delta_x = jnp.linspace(0, 1, w, retstep=True, endpoint=False)
        ys, delta_y = jnp.linspace(0, 1, h, retstep=True, endpoint=False)
        delta = jnp.array([delta_x, delta_y])

        offsets = jax.random.uniform(rng, (8, 2))

        def fit_cell(
            start: Float[Array, "2"], end: Float[Array, "2"]
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            points = jax.vmap(lambda offset: start + offset * (end - start))(offsets)

            samples = jax.vmap(lambda x: f(self, x))(points)

            return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)

        grid, approx_errors = jax.lax.map(
            lambda x: jax.lax.map(
                lambda y: fit_cell(jnp.array([x, y]), jnp.array([x, y]) + delta),
                ys,
                batch_size=BATCH_SIZE,
            ),
            xs,
            batch_size=BATCH_SIZE,
        )
        norm2 = jnp.sum(grid**2)
        rel_error2 = jnp.sum(approx_errors) / norm2

        return Representation(grid=grid), jnp.sqrt(rel_error2)


N_EPOCHS = args.n_epochs
LR = args.lr


ref_points = jax.random.uniform(jax.random.key(0), (20, 2))


def loss(f):
    return 0.5 * jnp.mean((jax.vmap(f)(ref_points) - jax.vmap(f_star)(ref_points)) ** 2)


def loss_grad(f, x):
    def kernel(a, b):
        d = 2
        SIGMA = 0.1
        return (
            (2 * jnp.pi) ** (-d / 2)
            * (SIGMA**d) ** (-1 / 2)
            * jnp.exp(-0.5 * ((a - b) @ (a - b)) / SIGMA)
        )

    return f(x) - f_star(x)


representation = Representation(grid=jnp.empty((args.grid_size, args.grid_size)))
representation(jnp.empty(2))
representation + representation
representation - representation
representation * jnp.empty(())
jnp.empty(()) * representation
representation.approx(loss_grad, rng=jax.random.key(0))


f = Representation.zeros(grid_size=args.grid_size)
f_history = []
rng = jax.random.key(0)
before = time.time()
for epoch_num in range(N_EPOCHS):
    print()
    print(f"Epoch #{epoch_num}")
    print()

    ic(f.grid.shape)

    this_rng, rng = jax.random.split(rng)
    g, rel_error = f.approx(loss_grad, rng=this_rng)
    ic(rel_error)

    f = f - LR * g
    jax.block_until_ready(f)
    f_history.append((f, time.time()))


for i, (f, this_time) in enumerate(f_history):
    xx, yy = jnp.meshgrid(jnp.linspace(0, 1, 1024), jnp.linspace(0, 1, 1024))
    result = jax.vmap(jax.vmap(f))(jnp.stack((xx, yy), axis=-1))
    np.save(f"{args.output}/{i:06}", result)

with open(f"{args.output}/timings.json", "w") as file:
    json.dump([float(this_time - before) for _f, this_time in f_history], file)
