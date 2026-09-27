import json
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import optax
from icecream import ic
from imageio import imread
from jaxtyping import Array, Float
from tqdm import tqdm

from src.teaser.common import f_star

BATCH_SIZE = 1024

# For Fourier embedding:
F = 0.75 * jax.random.normal(jax.random.key(0), (128, 2))


class MLP(nn.Module):
    layers: list[int]

    @nn.compact
    def __call__(self, x: Float[Array, "2"]) -> Float[Array, ""]:
        x = jnp.concatenate(
            [
                jnp.cos(2 * jnp.pi * F @ x),
                jnp.sin(2 * jnp.pi * F @ x),
            ]
        )
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = jax.nn.gelu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x)


N_EPOCHS = 2_000

mlp = MLP(layers=[256, 256, 256])


@jax.jit
@jax.value_and_grad
def loss(params, *, rng):
    samples = jax.random.uniform(rng, (BATCH_SIZE, 2))
    preds = jax.vmap(lambda x: mlp.apply(params, x))(samples)
    true_values = jax.vmap(f_star)(samples)
    return 0.5 * jnp.mean((preds - true_values) ** 2)


rng = jax.random.key(0)
params = mlp.init(jax.random.key(0), jnp.empty(2))
optimizer = optax.adam(
    learning_rate=3e-3 * 0.05
)  # * 0.05 to make it fair against ours, since we are artifically reducing the LR there
optimizer_state = optimizer.init(params)

loss(params, rng=jax.random.key(0))  # precompile

params_history = []
before = time.time()
for epoch_num in range(N_EPOCHS):
    print()
    print(f"Epoch #{epoch_num}")
    print()

    this_rng, rng = jax.random.split(rng)
    loss_value, loss_grad = loss(params, rng=this_rng)
    print(f"{loss_value = }")

    updates, optimizer_state = optimizer.update(loss_grad, optimizer_state)
    params = optax.apply_updates(params, updates)

    if epoch_num % 4 == 1:
        params_history.append((params, time.time()))
print(f"Done in {time.time() - before} seconds")


for i, (params, this_time) in enumerate(tqdm(params_history)):
    xx, yy = jnp.meshgrid(jnp.linspace(0, 1, 1024), jnp.linspace(0, 1, 1024))
    result = jax.vmap(jax.vmap(lambda x: mlp.apply(params, x)))(
        jnp.stack((xx, yy), axis=-1)
    )
    np.save(f"out-teaser-nn/{i:06}", result)

with open("out-teaser-nn/timings.json", "w") as file:
    json.dump(
        [float(this_time - before) for _params, this_time in params_history], file
    )
