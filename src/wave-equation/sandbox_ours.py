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


# BATCH_SIZE = 32
BATCH_SIZE = 64


@jdc.pytree_dataclass
class Representation:
    grid: Complex[Array, "t w h"]
    grid_sumt: Complex[Array, "w h"]
    grid_sumt_partial: Complex[Array, "w h"]
    bounds: jdc.Static[tuple[float, float]]

    @jax.jit
    def __call__(self, x):
        assert x.shape == (3,)
        nt, nx, ny = self.grid.shape
        a, b = self.bounds

        x = (x - a) / (b - a)
        i = jnp.floor(x[0] * nt).astype(jnp.int32)
        j = jnp.floor(x[1] * nx).astype(jnp.int32)
        k = jnp.floor(x[2] * ny).astype(jnp.int32)

        return jnp.where(
            (((0 <= i) & (i < nt)) & ((0 <= j) & (j < nx)) & ((0 <= k) & (k < ny))),
            self.grid[i, j, k],
            0.0,
        )

    @jax.jit
    def integral_t(self, x):
        assert x.shape == (2,)
        nt, nx, ny = self.grid.shape
        a, b = self.bounds

        x = (x - a) / (b - a)
        j = jnp.clip(jnp.floor(x[0] * nx).astype(jnp.int32), 0, nx - 1)
        k = jnp.clip(jnp.floor(x[1] * ny).astype(jnp.int32), 0, ny - 1)

        return jnp.where(
            (((0 <= j) & (j < nx)) & ((0 <= k) & (k < ny))),
            self.grid_sumt[j, k],
            0.0,
        )

    @jax.jit
    def integral_t_partial(self, x):
        assert x.shape == (2,)
        nt, nx, ny = self.grid.shape
        a, b = self.bounds

        x = (x - a) / (b - a)
        j = jnp.clip(jnp.floor(x[0] * nx).astype(jnp.int32), 0, nx - 1)
        k = jnp.clip(jnp.floor(x[1] * ny).astype(jnp.int32), 0, ny - 1)

        return jnp.where(
            (((0 <= j) & (j < nx)) & ((0 <= k) & (k < ny))),
            self.grid_sumt_partial[j, k],
            0.0,
        )

    @jax.jit
    def evaluate_spatial_grid(
        self,
        ts: Float[Array, "Mt"],
        xs: Float[Array, "Mx"],
        ys: Float[Array, "My"],
    ) -> Float[Array, "Mt Mx My"]:
        """
        Evaluates the exact continuous spatial function over a uniform grid.

        Args:
            ts, xs, ys: 1D arrays defining the spatial evaluation grid.
            freq_bounds: The (min, max) frequency bounds for the (t, x, y) axes
                         that the piecewise constant grid covers.
        """
        nt, nx, ny = self.grid.shape
        a, b = self.bounds

        # 1. Define the pixel widths based on the bounds
        du = (b - a) / nt
        dv = (b - a) / nx
        dw = (b - a) / ny

        # 2. Calculate the frequency centers of each bin
        u = a + (jnp.arange(nt) + 0.5) * du
        v = a + (jnp.arange(nx) + 0.5) * dv
        w = a + (jnp.arange(ny) + 0.5) * dw

        # 3. Compute 2D phase matrices (Frequency x Spatial)
        phase_u = jnp.exp(2j * jnp.pi * jnp.outer(u, ts))  # shape: (nt, Mt)
        phase_v = jnp.exp(2j * jnp.pi * jnp.outer(v, xs))  # shape: (nx, Mx)
        phase_w = jnp.exp(2j * jnp.pi * jnp.outer(w, ys))  # shape: (ny, My)

        # 4. Compute the separable 1D envelopes
        # The envelope scale depends strictly on the bin widths
        env_t = du * jnp.sinc(du * ts)
        env_x = dv * jnp.sinc(dv * xs)
        env_y = dw * jnp.sinc(dw * ys)

        summed_grid = jnp.einsum(
            "ijk,im,jn,kp->mnp", self.grid, phase_u, phase_v, phase_w
        )
        summed_grid = summed_grid * env_t[:, None, None]
        summed_grid = summed_grid * env_x[None, :, None]
        summed_grid = summed_grid * env_y[None, None, :]

        # 7. Extract the real component to discard floating-point noise
        return jnp.real(summed_grid)

    @classmethod
    def zeros(cls, bounds: tuple[float, float]):
        return Representation(
            grid=jnp.zeros((2, 2, 2), dtype=jnp.complex64),
            grid_sumt=jnp.zeros((2, 2), dtype=jnp.complex64),
            grid_sumt_partial=jnp.zeros((2, 2), dtype=jnp.complex64),
            bounds=bounds,
        )

    def subdivide(self) -> "Representation":
        return Representation(
            grid=jnp.repeat(
                jnp.repeat(jnp.repeat(self.grid, 2, axis=0), 2, axis=1), 2, axis=2
            ),
            grid_sumt=jnp.repeat(jnp.repeat(self.grid_sumt, 2, axis=0), 2, axis=1),
            grid_sumt_partial=jnp.repeat(
                jnp.repeat(self.grid_sumt_partial, 2, axis=0), 2, axis=1
            ),
            bounds=self.bounds,
        )

    @jax.jit
    def __add__(self, other: "Representation"):
        return Representation(
            grid=self.grid + other.grid,
            grid_sumt=self.grid_sumt + other.grid_sumt,
            grid_sumt_partial=self.grid_sumt_partial + other.grid_sumt_partial,
            bounds=self.bounds,
        )

    @jax.jit
    def __sub__(self, other: "Representation"):
        return Representation(
            grid=self.grid - other.grid,
            grid_sumt=self.grid_sumt - other.grid_sumt,
            grid_sumt_partial=self.grid_sumt_partial - other.grid_sumt_partial,
            bounds=self.bounds,
        )

    @jax.jit
    def __mul__(self, other: float):
        return Representation(
            grid=other * self.grid,
            grid_sumt=other * self.grid_sumt,
            grid_sumt_partial=other * self.grid_sumt_partial,
            bounds=self.bounds,
        )

    @jax.jit
    def __rmul__(self, other: float):
        return Representation(
            grid=self.grid * other,
            grid_sumt=self.grid_sumt * other,
            grid_sumt_partial=self.grid_sumt_partial * other,
            bounds=self.bounds,
        )

    @partial(jax.jit, static_argnames=("f",))
    def approx(
        self,
        f: Callable[["Representation", Float[Array, "3"]], Complex[Array, ""]],
        *,
        rng,
    ) -> tuple["Representation", float]:
        print("jitting")

        nt, nx, ny = self.grid.shape
        a, b = self.bounds

        # First, fit the grid:
        ts, delta_t = jnp.linspace(a, b, nt, retstep=True, endpoint=False)
        xs, delta_x = jnp.linspace(a, b, nx, retstep=True, endpoint=False)
        ys, delta_y = jnp.linspace(a, b, ny, retstep=True, endpoint=False)
        ts = ts + delta_t * 0.5
        xs = xs + delta_x * 0.5
        ys = ys + delta_y * 0.5

        grid = jax.lax.map(
            lambda t: jax.lax.map(
                lambda x: jax.lax.map(
                    lambda y: f(self, jnp.array([t, x, y])),
                    ys,
                    batch_size=BATCH_SIZE,
                ),
                xs,
                batch_size=BATCH_SIZE,
            ),
            ts,
            batch_size=BATCH_SIZE,
        )
        grid_sumt = jnp.sum(grid, axis=0) * delta_t
        t_partition = jnp.linspace(a, b, grid.shape[0] + 1)
        grid_sumt_partial = jnp.sum(
            grid
            * (0.5 * t_partition[1:] ** 2 - 0.5 * t_partition[:-1] ** 2)[:, None, None],
            axis=0,
        )

        out = Representation(
            grid=grid,
            grid_sumt=grid_sumt,
            grid_sumt_partial=grid_sumt_partial,
            bounds=self.bounds,
        )

        # Now, estimate the relative error:
        ts, delta_t = jnp.linspace(a, b, nt, retstep=True, endpoint=False)
        xs, delta_x = jnp.linspace(a, b, nx, retstep=True, endpoint=False)
        ys, delta_y = jnp.linspace(a, b, ny, retstep=True, endpoint=False)
        delta = jnp.array([delta_t, delta_x, delta_y])

        offsets = jax.random.uniform(rng, (8, 3))

        def evaluate(
            start: Float[Array, "3"], end: Float[Array, "3"]
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            points = jax.vmap(lambda offset: start + offset * (end - start))(offsets)
            samples = jax.vmap(lambda x: f(self, x))(points)
            fits = jax.vmap(out)(points)
            sobolev_symbol = jax.vmap(lambda p: 1 + (p @ p) + (p @ p) ** 2)(points)
            volume = jnp.prod(end - start)

            return (
                volume * jnp.mean(sobolev_symbol * jnp.abs(fits) ** 2),
                volume * jnp.mean(sobolev_symbol * jnp.abs(fits - samples) ** 2),
            )

        norm2s, approx_error2s = jax.lax.map(
            lambda t: jax.lax.map(
                lambda x: jax.lax.map(
                    lambda y: evaluate(
                        jnp.array([t, x, y]), jnp.array([t, x, y]) + delta
                    ),
                    ys,
                    batch_size=BATCH_SIZE,
                ),
                xs,
                batch_size=BATCH_SIZE,
            ),
            ts,
            batch_size=BATCH_SIZE,
        )
        rel_error2 = jnp.sum(approx_error2s) / jnp.sum(norm2s)

        return out, jnp.sqrt(rel_error2)


# N_EPOCHS = 50
N_EPOCHS = 20_000
LR = 0.1
# EPS = 0.5
# EPS = 2.0
EPS = 50.0

MU = 2
SCALE = 2

POSITIONS = jax.random.uniform(jax.random.key(0), (3, 2), minval=-3, maxval=+3)


def f_boundary_fourier(xy):
    def _bspline(t):
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

    x, y = xy / SCALE
    return (
        jnp.sum(
            jax.vmap(
                lambda position: jnp.exp(2j * jnp.pi * position[0] * x)
                * jnp.exp(2j * jnp.pi * position[1] * y)
                * _bspline(x)
                * _bspline(y)
            )(POSITIONS)
        )
        / SCALE**2
    )


def f_boundary(xy):
    x, y = xy * SCALE
    return jnp.sum(
        jax.vmap(
            lambda position: jnp.sinc(x + position[0]) ** 3
            * jnp.sinc(y + position[1]) ** 3
        )(POSITIONS)
    )


def loss(f_fourier):
    return jnp.nan  # TODO


def loss_grad_fourier(f_fourier, omega):
    tau, xi, zeta = omega
    # pde_term = 16 * jnp.pi**4 * (tau**2 - MU**2 * (xi**2 + zeta**2)) ** 2 * f_fourier(omega)
    # boundary_term1 = f_fourier.integral_t(omega[1:]) - f_boundary_fourier(omega[1:])
    # boundary_term2 = 4 * jnp.pi**2 * tau * f_fourier.integral_t_partial(omega[1:])
    pde_term = (tau**2 - MU**2 * (xi**2 + zeta**2)) ** 2 * f_fourier(omega)
    boundary_term1 = f_fourier.integral_t(omega[1:]) - f_boundary_fourier(omega[1:])
    boundary_term2 = tau * f_fourier.integral_t_partial(omega[1:])

    out = (pde_term + boundary_term1 + boundary_term2) / (
        # 1 + 4 * jnp.pi**2 * (tau**2 + xi**2 + zeta**2) + 16 * jnp.pi**4 * (tau**2 + xi**2 + zeta**2) ** 2
        # 1 + (tau**2 + xi**2 + zeta**2) + (tau**2 + xi**2 + zeta**2) ** 2
        (1 + (jnp.abs(tau) ** 2 + jnp.abs(xi) ** 2 + jnp.abs(zeta) ** 2)) ** 2
    )
    return out


def true_solution(txy):
    t, x, y = txy
    c = MU
    res = 10

    def _bspline(t):
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

    def integral_at_t(time):
        # Grid for the disk of dependence
        phi = jnp.linspace(0, jnp.pi / 2, res)
        theta = jnp.linspace(0, 2 * jnp.pi, res)
        p_grid, t_grid = jnp.meshgrid(phi, theta)

        r = c * time * jnp.sin(p_grid)
        xi = x + r * jnp.cos(t_grid)
        eta = y + r * jnp.sin(t_grid)

        # Integration with the transformed kernel
        # The sin(phi) term handles the Poisson singularity
        vals = _bspline(xi) * _bspline(eta) * jnp.sin(p_grid)
        return (time / (2 * jnp.pi)) * jnp.trapezoid(
            jnp.trapezoid(vals, phi, axis=1), theta
        )

    # u(x, y, t) = d/dt [Integral]
    return jax.grad(integral_at_t)(t)


# ts = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

before = time.time()

rng = jax.random.key(0)
f_fourier = Representation.zeros(bounds=(-5, +5))
for _ in range(2):
    # for _ in range(4):
    f_fourier = f_fourier.subdivide()
assert jnp.all(f_fourier.grid == 0.0)
for epoch_num in range(N_EPOCHS):
    print()
    print(f"Epoch #{epoch_num}; loss = {loss(f_fourier)}")
    print()

    ic(f_fourier.grid.shape)

    this_rng, rng = jax.random.split(rng)
    g_fourier, rel_error = f_fourier.approx(loss_grad_fourier, rng=this_rng)
    ic(rel_error)
    while rel_error > EPS and f_fourier.grid.shape[0] < 128:
        print("subdividing")
        f_fourier = f_fourier.subdivide()
        g_fourier, rel_error = f_fourier.approx(loss_grad_fourier, rng=this_rng)
        ic(rel_error)

    f_fourier = f_fourier - LR * g_fourier

    if epoch_num % 1_000 != 1:
        continue  # skip plot

    if args.plot:
        f_grid = f_fourier.evaluate_spatial_grid(
            jnp.linspace(0, 1, 200),
            jnp.linspace(-3, +3, 200),
            jnp.linspace(-3, +3, 200),
        )

        fig, axs = plt.subplots(2, len(ts), figsize=(10.4, 4.8))

        for ax in np.ravel(axs):
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0, 0].set_ylabel("Fit")
        axs[1, 0].set_ylabel("True")

        for t_index, t in enumerate(ts):
            xx, yy = jnp.meshgrid(jnp.linspace(-3, +3, 200), jnp.linspace(-3, +3, 200))
            tt = jnp.ones_like(xx) * t

            axs[0, t_index].set_title(f"t = ${t}$")

            def minmax(arr):
                ic(jnp.min(arr), jnp.max(arr))
                return arr

            # Estimate:

            axs[0, t_index].imshow(
                f_grid[
                    jnp.around((t / 2) * (f_grid.shape[0] - 1)).astype(jnp.int32),
                    :,
                    :,
                ].T,
                clim=(-0.20, 0.20),
                cmap="RdBu_r",
            )
            # axs[1, t_index].imshow(
            #     minmax(
            #         jax.vmap(jax.vmap(true_solution))(jnp.stack((tt, xx, yy), axis=-1))[
            #             :, :, 0
            #         ]
            #     ),
            #     clim=(0.1, 0.3),
            # )

        plt.savefig(f"out2/{epoch_num:09}.png")
        plt.close()

end = time.time()

out = np.empty((len(ts), 200, 200))

f_grid = f_fourier.evaluate_spatial_grid(
    jnp.linspace(0, 2, 200),
    jnp.linspace(-3, +3, 200),
    jnp.linspace(-3, +3, 200),
)
for t_index, t in enumerate(ts):
    xx, yy = jnp.meshgrid(jnp.linspace(-3, +3, 200), jnp.linspace(-3, +3, 200))
    tt = jnp.ones_like(xx) * t

    out[t_index, :, :] = f_grid[
        jnp.around((t / 2) * (f_grid.shape[0] - 1)).astype(jnp.int32), :, :
    ].T

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
