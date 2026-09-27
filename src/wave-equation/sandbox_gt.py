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
args = parser.parse_args()


MU = 2
SCALE = 2
POSITIONS = jax.random.uniform(jax.random.key(0), (3, 2), minval=-3, maxval=+3)
A, B = -3, +3
GRID_BOUNDS = -5, +5


def f_boundary(xy):
    x, y = xy * SCALE
    return jnp.sum(
        jax.vmap(
            lambda position: jnp.sinc(x + position[0]) ** 3
            * jnp.sinc(y + position[1]) ** 3
        )(POSITIONS)
    )


def simulate_2d_wave_jax(grid_size=1_000, grid_size_T=1_000, T=1.0):
    # 1. Calculate dx from the spatial bounds
    x_min, x_max = GRID_BOUNDS
    dx = (x_max - x_min) / (grid_size - 1)

    # # 2. Automatically determine the optimal dt using the CFL stability condition.
    # # The max stable Courant number for 2D is 1/sqrt(2). We use 0.99 for a safety margin.
    # courant_number = (1.0 / jnp.sqrt(2.0)) * 0.99
    # dt = courant_number * dx / MU
    # C_sq = courant_number**2
    dt = T / grid_size_T
    C_sq = (MU * dt / dx) ** 2

    steps = int(jnp.ceil(T / dt))

    # 3. Coordinate grid using linspace
    axis = jnp.linspace(x_min, x_max, grid_size)
    xx, yy = jnp.meshgrid(axis, axis)

    # 4. The Update Step (JIT compiled for massive speedup)
    @jax.jit
    def update_step(u_c, u_p):
        # Calculate the interior points (creating a smaller array)
        interior = (
            2 * u_c[1:-1, 1:-1]
            - u_p[1:-1, 1:-1]
            + C_sq
            * (
                u_c[2:, 1:-1]
                + u_c[:-2, 1:-1]
                + u_c[1:-1, 2:]
                + u_c[1:-1, :-2]
                - 4 * u_c[1:-1, 1:-1]
            )
        )

        # Pad the interior with a border of 1 zero on all sides to maintain boundaries
        return jnp.pad(interior, pad_width=1, mode="constant", constant_values=0.0)

    # Initialize states
    grid = jnp.zeros((steps + 1, grid_size, grid_size))

    xy_stack = jnp.stack((xx, yy), axis=-1)
    grid = grid.at[0, :, :].set(jax.vmap(jax.vmap(f_boundary))(xy_stack))

    @jax.jit
    def first_step_fix(u0):
        laplacian = (
            u0[2:, 1:-1]
            + u0[:-2, 1:-1]
            + u0[1:-1, 2:]
            + u0[1:-1, :-2]
            - 4 * u0[1:-1, 1:-1]
        )
        # Note the 0.5 multiplier on the C_sq term:
        interior = u0[1:-1, 1:-1] + 0.5 * C_sq * laplacian
        return jnp.pad(interior, pad_width=1, mode="constant", constant_values=0.0)

    grid = grid.at[1].set(first_step_fix(grid[0]))

    # 5. Main finite difference loop
    for t_index in range(1, steps):
        u_next = update_step(grid[t_index], grid[t_index - 1])
        grid = grid.at[t_index + 1].set(u_next)

    return grid


before = time.time()

# Run the JAX simulation
grid = simulate_2d_wave_jax()
final_state = grid[-1, :, :]

end = time.time()

ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Visualization
plt.figure(figsize=(7, 6))
extent = [-5, +5, -5, +5]
plt.imshow(final_state, cmap="RdBu", origin="lower", extent=extent, vmin=-0.5, vmax=0.5)
plt.title("2D Wave Equation (Pure JAX)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Wave Amplitude")
plt.savefig("out.png")

print(end - before)

start_i = int(
    ((A - GRID_BOUNDS[0]) / (GRID_BOUNDS[1] - GRID_BOUNDS[0])) * grid.shape[1]
)
end_i = int(((B - GRID_BOUNDS[0]) / (GRID_BOUNDS[1] - GRID_BOUNDS[0])) * grid.shape[1])
out = np.empty((len(ts), end_i - start_i, end_i - start_i))

for t_index, t in enumerate(ts):
    t_index_in_grid = jnp.clip(
        jnp.around(t * (grid.shape[0] - 1)), 0, grid.shape[0] - 1
    ).astype(jnp.int32)

    out[t_index, :, :] = grid[t_index_in_grid, start_i:end_i, start_i:end_i]

np.save(f"{args.output}/results.npy", out)
with open(f"{args.output}/data.json", "w") as file:
    json.dump(
        {
            "training time": end - before,
            "ts": ts,
        },
        file,
    )
