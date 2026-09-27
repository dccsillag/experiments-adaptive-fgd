import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.signal as jsp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import optax
from icecream import ic
from imageio import imread
from jaxtyping import Array, Complex, Float

parser = ArgumentParser()
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--eval_every", type=int, default=200)
args = parser.parse_args()


BATCH_SIZE = 4096

MU = 2
SCALE = 2
POSITIONS = jax.random.uniform(jax.random.key(0), (3, 2), minval=-3, maxval=+3)
A, B = -3, +3


def f_boundary(xy):
    x, y = xy * SCALE
    return jnp.sum(
        jax.vmap(
            lambda position: jnp.sinc(x + position[0]) ** 3
            * jnp.sinc(y + position[1]) ** 3
        )(POSITIONS)
    )


def loss(f, *, rng):
    # PDE term
    this_rng, rng = jax.random.split(rng)
    points = jax.random.uniform(
        this_rng, (BATCH_SIZE, 3), minval=-3, maxval=+3
    )  # FIXME bounds
    points = points.at[:, 0].set((points[:, 0] + 3) / 6)
    partials2 = jax.vmap(lambda p: jnp.diag(jax.hessian(f)(p)))(points)
    pde_term = 0.5 * jnp.mean(
        (partials2[:, 0] - MU**2 * (partials2[:, 1] + partials2[:, 2])) ** 2
    )

    # Boundary 1
    this_rng, rng = jax.random.split(rng)
    points = jax.random.uniform(this_rng, (BATCH_SIZE, 3), minval=-3, maxval=+3)
    points = points.at[:, 0].set(0.0)
    values = jax.vmap(f)(points)
    boundary1_term = 0.5 * jnp.mean((values - jax.vmap(f_boundary)(points[:, 1:])) ** 2)

    # Boundary 2
    this_rng, rng = jax.random.split(rng)
    points = jax.random.uniform(this_rng, (BATCH_SIZE, 3), minval=-3, maxval=+3)
    points = points.at[:, 0].set(0.0)
    partials = jax.vmap(jax.grad(f))(points)
    boundary2_term = 0.5 * jnp.mean((partials[:, 0]) ** 2)

    return pde_term + boundary1_term + boundary2_term


F = (1 / 3) * jax.random.normal(jax.random.key(0), (128, 3))


class MLP(nn.Module):
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.concatenate(
            [
                jnp.cos(2 * jnp.pi * F @ x),
                jnp.sin(2 * jnp.pi * F @ x),
            ]
        )
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = jax.nn.gelu(x)
        out = nn.Dense(1)(x)
        return jnp.squeeze(out)


mlp = MLP([256, 256, 256, 256, 256])


@jax.jit
@jax.value_and_grad
def nn_loss(params, *, rng):
    return loss(lambda x: mlp.apply(params, x), rng=rng)


rng = jax.random.key(0)

ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_EPOCHS = 100_000

before = time.time()

this_rng, rng = jax.random.split(rng)
params = mlp.init(this_rng, jnp.zeros(3))
optimizer = optax.adam(learning_rate=1e-4)
optimizer_state = optimizer.init(params)
for epoch_num in range(N_EPOCHS):
    this_rng, rng = jax.random.split(rng)
    loss_value, loss_grads = nn_loss(params, rng=this_rng)

    print()
    print(f"Epoch #{epoch_num}; loss = {loss_value}")
    print()

    updates, optimizer_state = optimizer.update(loss_grads, optimizer_state)
    params = optax.apply_updates(params, updates)

    if epoch_num % args.eval_every != 1:
        continue  # skip plot

    f = lambda x: mlp.apply(params, x)

    if args.plot:
        fig, axs = plt.subplots(1, len(ts), figsize=(10.4, 3.4))

        for t_index, t in enumerate(ts):
            xx, yy = jnp.meshgrid(jnp.linspace(A, B, 200), jnp.linspace(A, B, 200))
            tt = jnp.ones_like(xx) * t

            axs[t_index].imshow(
                jax.vmap(jax.vmap(f))(jnp.stack((tt, xx, yy), axis=-1)),
                clim=(-0.20, +0.20),
                cmap="RdBu_r",
            )
            axs[t_index].axis("off")

        plt.savefig(f"{args.output}/plot{epoch_num:09}.png")
        plt.close()

end = time.time()

out = np.empty((len(ts), 200, 200))

for t_index, t in enumerate(ts):
    xx, yy = jnp.meshgrid(jnp.linspace(A, B, 200), jnp.linspace(A, B, 200))
    tt = jnp.ones_like(xx) * t

    out[t_index, :, :] = jax.vmap(jax.vmap(f))(jnp.stack((tt, xx, yy), axis=-1))

np.save(f"{args.output}/results.npy", out)
with open(f"{args.output}/data.json", "w") as file:
    json.dump(
        {
            "training time": end - before,
            "number of epochs": N_EPOCHS,
            "ts": ts,
        },
        file,
    )
