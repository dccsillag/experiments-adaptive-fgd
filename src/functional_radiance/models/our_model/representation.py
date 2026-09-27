import ctypes
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Protocol, Self

import dill
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse
from icecream import ic
from jax.experimental import checkify
from jaxtyping import Array, Bool, Float, Int, PyTree
from tqdm import tqdm

from src.functional_radiance.models.our_model.regularization import (
    DensityRegularization,
)
from src.functional_radiance.utils.data import Dataset
from src.functional_radiance.utils.general import (
    BaseLearner,
    JaxRng,
    cumulative_trapezoid,
    jax_predict,
    repr_lambda,
    viewing_direction_to_unit_circle,
)
from src.functional_radiance.utils.models import Camera, CameraAndImage, Ray
from src.functional_radiance.utils.sh_utils import C0, eval_sh

so_file = "bin/render.so"
if os.path.exists(so_file):
    render_lib = ctypes.cdll.LoadLibrary(so_file)
    jax.ffi.register_ffi_target(
        "render_within_interval_cuda",
        jax.ffi.pycapsule(render_lib.render_within_interval_cuda),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "render_within_interval_trilerp_cuda",
        jax.ffi.pycapsule(render_lib.render_within_interval_trilerp_cuda),
        platform="CUDA",
    )
    CUDA_RENDER = True
else:
    warnings.warn(
        f"CUDA renderer for functional radiance is not compiled! Using the pure JAX renderer. To compile, run `make` from the project root."
    )
    CUDA_RENDER = False


class Representation(Protocol):
    def get_hparams_json(self) -> dict:
        pass

    def init(self) -> PyTree:
        pass

    def add(
        self, state_a: PyTree, state_b: PyTree, *, assume_compatible: bool
    ) -> PyTree:
        pass

    def scale(self, scalar: float, state: PyTree) -> PyTree:
        pass

    def scale_density(self, scalar: float, state: PyTree) -> PyTree:
        pass

    def scale_color(self, scalar: float, state: PyTree) -> PyTree:
        pass

    def eval_density(self, state: PyTree, x):
        pass

    def eval_color(self, state: PyTree, x, direction):
        pass

    def density_norm(self, state: PyTree) -> Float[Array, ""]:
        pass

    def color_norm(self, state: PyTree) -> Float[Array, ""]:
        pass

    def clamp_density(self, state: PyTree, bounds: tuple[float, float]) -> PyTree:
        pass

    def approx_gradient(
        self,
        state: PyTree,
        functional_gradient: Callable[
            [Float[Array, "d"], Float[Array, "k d"]],
            tuple[Float[Array, ""], Float[Array, "k 3"]],
        ],
        data: Dataset,
        *,
        rng,
    ) -> tuple[PyTree, Float[Array, ""]]:  # (representation_state, rel_error)
        pass

    def subdivide(self, state: PyTree) -> PyTree:
        pass

    def render_within_interval(
        self,
        state: PyTree,
        *,
        ray: Ray,
        t_start: float,
        t_end: float,
    ) -> tuple[
        Float[Array, "3"],  # rendered color
        Float[Array, ""],  # transmittance at t_end
    ]:
        pass


@dataclass
class ProperVoxelGridRepresentation(Representation):
    init_grid_size: int
    bounds_x: tuple[float, float]
    bounds_y: tuple[float, float]
    bounds_z: tuple[float, float]
    initial_density: Float[Array, ""]
    initial_color: Float[Array, "3"]
    sh_deg: int
    batch_size: int
    n_random_directions: int | None  # None => directions connecting to the camera
    n_samples_per_voxel: int
    sampling_pattern: Literal["naive"] | Literal["symmetrical"] | Literal["corners"]
    density_trilinear_interp: bool

    def get_hparams_json(self) -> dict:
        return {
            "representation": "proper_voxel_grid",
            "init_grid_size": repr(self.init_grid_size),
            "bounds_x": repr(self.bounds_x),
            "bounds_y": repr(self.bounds_y),
            "bounds_z": repr(self.bounds_z),
            "initial_density": repr(self.initial_density),
            "initial_color": repr(self.initial_color),
            "sh_deg": self.sh_deg,
            "batch_size": self.batch_size,
            "n_random_directions": self.n_random_directions,
            "n_samples_per_voxel": self.n_samples_per_voxel,
            "sampling_pattern": self.sampling_pattern,
            "density_trilinear_interp": self.density_trilinear_interp,
        }

    def init(self):
        initial_sh_coeffs = (
            jnp.zeros((3, (self.sh_deg + 1) ** 2)).at[:, 0].set(self.initial_color / C0)
        )
        assert jnp.all(
            jnp.isclose(
                eval_sh(
                    self.sh_deg,
                    initial_sh_coeffs[None, :],
                    jnp.array([[1.0, 0.0, 0.0]]),
                ),
                self.initial_color[None],
            )
        )

        if self.density_trilinear_interp:
            density_grid = (
                jnp.ones(
                    (self.init_grid_size, self.init_grid_size, self.init_grid_size, 8)
                )
                * self.initial_density
            )
        else:
            density_grid = (
                jnp.ones(
                    (self.init_grid_size, self.init_grid_size, self.init_grid_size)
                )
                * self.initial_density
            )
        color_grid = (
            jnp.ones(
                (
                    self.init_grid_size,
                    self.init_grid_size,
                    self.init_grid_size,
                    3,
                    (self.sh_deg + 1) ** 2,
                )
            )
            * initial_sh_coeffs[None, None, None, :, :]
        )

        return {
            "density": density_grid,
            "color": color_grid,
        }

    def add(
        self, state_a: PyTree, state_b: PyTree, *, assume_compatible: bool
    ) -> PyTree:
        return {
            "density": state_a["density"] + state_b["density"],
            "color": state_a["color"] + state_b["color"],
        }

    def scale(self, scalar: float, state: PyTree) -> PyTree:
        return {
            "density": scalar * state["density"],
            "color": scalar * state["color"],
        }

    def scale_density(self, scalar: float, state: PyTree) -> PyTree:
        return {
            "density": scalar * state["density"],
            "color": state["color"],
        }

    def scale_color(self, scalar: float, state: PyTree) -> PyTree:
        return {
            "density": state["density"],
            "color": scalar * state["color"],
        }

    def eval_density(self, state, x):
        i, j, k, ti, tj, tk, inside = self._interp_to_grid(state, x)
        if self.density_trilinear_interp:
            out = _trilinear_interp(state["density"][i, j, k], ti, tj, tk)
        else:
            out = state["density"][i, j, k]
        return jnp.where(inside, out, 0.0)

    def eval_color(self, state, x, direction):
        i, j, k, ti, tj, tk, inside = self._interp_to_grid(state, x)
        coeffs = state["color"][i, j, k]
        return jnp.where(
            inside,
            eval_sh(self.sh_deg, coeffs, direction),
            jnp.array([0.0, 0.0, 0.0]),
        )

    def density_norm(self, state: PyTree) -> Float[Array, ""]:
        voxel_size = (
            ((self.bounds_x[1] - self.bounds_x[0]) / state["density"].shape[0])
            * ((self.bounds_y[1] - self.bounds_y[0]) / state["density"].shape[1])
            * ((self.bounds_z[1] - self.bounds_z[0]) / state["density"].shape[2])
        )
        return jnp.sum(state["density"] ** 2 * voxel_size)

    def color_norm(self, state: PyTree) -> Float[Array, ""]:
        voxel_size = (
            ((self.bounds_x[1] - self.bounds_x[0]) / state["color"].shape[0])
            * ((self.bounds_y[1] - self.bounds_y[0]) / state["color"].shape[1])
            * ((self.bounds_z[1] - self.bounds_z[0]) / state["color"].shape[2])
        )
        return jnp.sum(
            jnp.sum(state["color"] ** 2, axis=(-1, -2)) * voxel_size
        )  # XXX double-check

    def clamp_density(self, state: PyTree, bounds: tuple[float, float]) -> PyTree:
        return {
            "density": jnp.clip(state["density"], *bounds),
            "color": state["color"],
        }

    def approx_gradient(
        self,
        state: PyTree,
        functional_gradient: Callable[
            [Float[Array, "d"], Float[Array, "k d"]],
            tuple[Float[Array, ""], Float[Array, "k 3"]],
        ],
        data: Dataset,
        *,
        rng,
    ) -> tuple[PyTree, Float[Array, ""]]:
        # setup grid of locations
        left_linspace = lambda a, b, grid_size: (
            jnp.linspace(a, b, grid_size + 1)[:-1],
            (b - a) / grid_size,
        )
        xs, delta_x = left_linspace(*self.bounds_x, state["density"].shape[0])
        ys, delta_y = left_linspace(*self.bounds_y, state["density"].shape[1])
        zs, delta_z = left_linspace(*self.bounds_z, state["density"].shape[2])

        voxel_size = jnp.array([delta_x, delta_y, delta_z])

        # vmap over them, computing the functional gradients
        density_grid, color_grid, approx_errors = jax.lax.map(
            lambda x: jax.lax.map(
                lambda y: jax.lax.map(
                    lambda z: fit_voxel(
                        jnp.array([x, y, z]),
                        voxel_size,
                        sampling_pattern=self.sampling_pattern,
                        n_samples_per_voxel=self.n_samples_per_voxel,
                        n_random_directions=self.n_random_directions,
                        sh_deg=self.sh_deg,
                        density_trilinear_interp=self.density_trilinear_interp,
                        data=data,
                        functional_gradient=functional_gradient,
                        rng=rng,
                    ),
                    zs,
                    batch_size=self.batch_size,
                ),
                ys,
                batch_size=self.batch_size,
            ),
            xs,
            batch_size=self.batch_size,
        )

        return {
            "density": density_grid,
            "color": color_grid,
        }, jnp.sum(approx_errors)

    def subdivide(self, state: PyTree) -> PyTree:
        return {
            "density": jnp.repeat(
                jnp.repeat(jnp.repeat(state["density"], 2, axis=0), 2, axis=1),
                2,
                axis=2,
            ),
            "color": jnp.repeat(
                jnp.repeat(jnp.repeat(state["color"], 2, axis=0), 2, axis=1), 2, axis=2
            ),
        }

    def render_within_interval(
        self,
        state: PyTree,
        *,
        ray: Ray,
        t_start: float,
        t_end: float,
    ) -> tuple[
        Float[Array, "3"],  # rendered color
        Float[Array, ""],  # transmittance at t_end
    ]:
        if not CUDA_RENDER:
            raise RuntimeError(
                "exact rendering is only implemented in the CUDA backend, run `make`"
            )

        ni, nj, nk = state["density"].shape
        assert ni == nj == nk
        grid_size = ni

        if self.density_trilinear_interp:
            return jax.ffi.ffi_call(  # type: ignore
                "render_within_interval_trilerp_cuda",
                (
                    jax.ShapeDtypeStruct((3,), jnp.float32),
                    jax.ShapeDtypeStruct((), jnp.float32),
                ),
                vmap_method="expand_dims",
            )(
                jnp.concatenate(
                    (
                        ray.origin,
                        ray.direction,
                        jnp.array([t_start, t_end]),
                    ),
                    axis=-1,
                ).astype(jnp.float32),
                state["density"].astype(jnp.float32),
                state["color"].astype(jnp.float32),
                grid_size=grid_size,
                sh_deg=self.sh_deg,
                bbox_start_x=self.bounds_x[0],
                bbox_start_y=self.bounds_y[0],
                bbox_start_z=self.bounds_z[0],
                bbox_end_x=self.bounds_x[1],
                bbox_end_y=self.bounds_y[1],
                bbox_end_z=self.bounds_z[1],
            )
        else:
            return jax.ffi.ffi_call(  # type: ignore
                "render_within_interval_cuda",
                (
                    jax.ShapeDtypeStruct((3,), jnp.float32),
                    jax.ShapeDtypeStruct((), jnp.float32),
                ),
                vmap_method="expand_dims",
            )(
                jnp.concatenate(
                    (
                        ray.origin,
                        ray.direction,
                        jnp.array([t_start, t_end]),
                    ),
                    axis=-1,
                ).astype(jnp.float32),
                state["density"].astype(jnp.float32),
                state["color"].astype(jnp.float32),
                grid_size=grid_size,
                sh_deg=self.sh_deg,
                bbox_start_x=self.bounds_x[0],
                bbox_start_y=self.bounds_y[0],
                bbox_start_z=self.bounds_z[0],
                bbox_end_x=self.bounds_x[1],
                bbox_end_y=self.bounds_y[1],
                bbox_end_z=self.bounds_z[1],
            )

    def _interp_to_grid(
        self, state: PyTree, x: Float[Array, "3"]
    ) -> tuple[
        Int[Array, ""],
        Int[Array, ""],
        Int[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
        Bool[Array, ""],
    ]:
        ni, nj, nk = state["density"].shape
        i, ti, i_inside = self._interp_to_grid_on_axis(
            x[0], *self.bounds_x, grid_size=ni
        )
        j, tj, j_inside = self._interp_to_grid_on_axis(
            x[1], *self.bounds_y, grid_size=nj
        )
        k, tk, k_inside = self._interp_to_grid_on_axis(
            x[2], *self.bounds_z, grid_size=nk
        )
        return i, j, k, ti, tj, tk, i_inside & j_inside & k_inside

    def _interp_to_grid_on_axis(
        self, x: Float[Array, ""], inf: float, sup: float, *, grid_size: int
    ) -> tuple[Int[Array, ""], Float[Array, ""], Bool[Array, ""]]:
        t = (x - inf) / (sup - inf)
        index = jnp.floor(t * grid_size)
        index = jnp.clip(index.astype(jnp.int32), 0, grid_size - 1)
        fractional = t * grid_size - index
        fractional = jnp.clip(fractional, 0.0, 1.0)
        return index, fractional, (0.0 <= t) & (t <= 1.0)


def lstsq(a, b):
    # JAX's jnp.linalg.lstsq seems to occasionally give NaNs; cf. https://github.com/jax-ml/jax/issues/26468

    match a.shape:
        case ():
            return b / a
        case (_, _):
            # jax.config.update("jax_enable_x64", True)
            # out = (jnp.linalg.pinv(a.astype(jnp.float64)) @ b).astype(jnp.float32)
            # jax.config.update("jax_enable_x64", False)
            # return out
            return jnp.linalg.pinv(a) @ b

            # q, r = jnp.linalg.qr(a)
            # return jax.lax.linalg.triangular_solve(r, q.T @ b, lower=False)

            # u, s, vh = jnp.linalg.svd(a, full_matrices=False)
            # max_s = jnp.max(s)
            # s_inv = jnp.where(s > 1e-15 * max_s, 1.0 / s, 0.0)
            # res = vh.T @ (s_inv[:, None] * (u.T @ b))
            # ic(a.shape)
            # ic(b.shape)
            # ic(res.shape)
            # return res

            # x_sol, _, _, _ = jnp.linalg.lstsq(a, b, rcond=None)
            # return x_sol
        case _:
            raise TypeError("bad shape for lstsq")

    # q, r = jnp.linalg.qr(a)
    # z = q.T @ b
    # return jax.lax.linalg.triangular_solve(r, z, left_side=True)

    # # Form normal equations: A^T A x = A^T b
    # ata = a.T @ a
    # atb = a.T @ b
    #
    # # Solve using Cholesky decomposition
    # c, lower = jax.scipy.linalg.cho_factor(ata)
    # x = jax.scipy.linalg.cho_solve((c, lower), atb)
    #
    # return x


def fit_voxel(
    voxel_corner: Float[Array, "3"],
    voxel_size: Float[Array, "3"],
    *,
    sampling_pattern: Literal["naive"] | Literal["symmetrical"] | Literal["corners"],
    n_samples_per_voxel: int,
    n_random_directions: int | None,
    sh_deg: int,
    density_trilinear_interp: bool,
    data: Dataset,
    functional_gradient: Callable[
        [Float[Array, "d"], Float[Array, "k d"]],
        tuple[Float[Array, ""], Float[Array, "k 3"]],
    ],
    rng,
) -> tuple[
    Float[Array, "*density_coeffs"], Float[Array, "*color_coeffs"], Float[Array, ""]
]:
    match sampling_pattern:
        case "naive":
            offsets = (
                jax.random.uniform(rng, (n_samples_per_voxel, 3)) * voxel_size[None, :]
            )
        case "symmetrical":
            assert n_samples_per_voxel % 8 == 0, (
                "for the symmetrical ssampling scheme the number of samples should be a multiple of 8"
            )
            n_corner_samples = n_samples_per_voxel // 8
            offsets = jax.random.uniform(rng, (n_corner_samples, 3)) * (
                voxel_size[None, :] * 0.5
            )
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array([[voxel_size[0], 0, 0]])
                    + jnp.array([[-1, +1, +1]]) * offsets,
                ],
                axis=0,
            )
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array([[0, voxel_size[1], 0]])
                    + jnp.array([[+1, -1, +1]]) * offsets,
                ],
                axis=0,
            )
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array([[0, 0, voxel_size[2]]])
                    + jnp.array([[+1, +1, -1]]) * offsets,
                ],
                axis=0,
            )
            assert offsets.shape == (n_samples_per_voxel, 3)
        case "symmetrical+corners":
            assert n_samples_per_voxel % 8 == 0, (
                "for the symmetrical ssampling scheme the number of samples should be a multiple of 8"
            )
            n_interior_samples = n_samples_per_voxel - 8
            n_corner_samples = n_interior_samples // 8
            offsets = jax.random.uniform(rng, (n_corner_samples, 3)) * (
                voxel_size[None, :] * 0.5
            )
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array([[voxel_size[0], 0, 0]])
                    + jnp.array([[-1, +1, +1]]) * offsets,
                ],
                axis=0,
            )
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array([[0, voxel_size[1], 0]])
                    + jnp.array([[+1, -1, +1]]) * offsets,
                ],
                axis=0,
            )
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array([[0, 0, voxel_size[2]]])
                    + jnp.array([[+1, +1, -1]]) * offsets,
                ],
                axis=0,
            )
            assert offsets.shape == (n_interior_samples, 3)
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.array(
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, voxel_size[2]],
                            [0.0, voxel_size[1], 0.0],
                            [0.0, voxel_size[1], voxel_size[2]],
                            [voxel_size[0], 0.0, 0.0],
                            [voxel_size[0], 0.0, voxel_size[2]],
                            [voxel_size[0], voxel_size[1], 0.0],
                            [voxel_size[0], voxel_size[1], voxel_size[2]],
                        ]
                    ),
                ],
                axis=0,
            )
            assert offsets.shape == (n_samples_per_voxel, 3)
        case "corners":
            assert n_samples_per_voxel == 8, (
                "for the 'corners' ssampling scheme the number of samples should be exactly 8"
            )
            offsets = jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, voxel_size[2]],
                    [0.0, voxel_size[1], 0.0],
                    [0.0, voxel_size[1], voxel_size[2]],
                    [voxel_size[0], 0.0, 0.0],
                    [voxel_size[0], 0.0, voxel_size[2]],
                    [voxel_size[0], voxel_size[1], 0.0],
                    [voxel_size[0], voxel_size[1], voxel_size[2]],
                ]
            )
        case _:
            raise ValueError("unimplemented sampling pattern")

    density_loss = lambda coeffs, values, tijk: (
        (_trilinear_interp(coeffs, *tijk) - values) ** 2
        if density_trilinear_interp
        else 0.5 * (coeffs - values) ** 2
    )
    density_loss_grad = jax.grad(density_loss)
    density_loss_hessian = jax.hessian(density_loss)
    color_loss = lambda coeffs, values, dirs: jnp.mean(
        (eval_sh(sh_deg, coeffs[None, None, :], dirs)[:, 0] - values) ** 2
    ) * (4.0 * jnp.pi)
    density_coeffs0 = jnp.zeros(8) if density_trilinear_interp else jnp.array(0.0)
    color_loss_grad = jax.grad(color_loss)
    color_loss_hessian = jax.hessian(color_loss)
    color_coeffs0 = jnp.zeros((sh_deg + 1) ** 2)

    def process_single_offset(offset):
        point = voxel_corner + offset

        if n_random_directions is None:
            directions = data.map(
                lambda datum: datum.camera.ray_at_normalized(
                    datum.camera.world_position_to_uv(point)
                ).direction,
                batch_size=None,
            )
        else:
            directions = jax.random.normal(jax.random.key(0), (n_random_directions, 3))
            directions = directions / jnp.linalg.norm(directions, axis=1)[:, None]

        density_gradient, color_gradients = functional_gradient(point, directions)

        tijk = offset / voxel_size  # XXX check
        density_value = density_loss(density_coeffs0, density_gradient, tijk)
        density_grad = density_loss_grad(density_coeffs0, density_gradient, tijk)
        density_hessian = density_loss_hessian(density_coeffs0, density_gradient, tijk)

        color_value = jax.vmap(
            lambda color_gradients_c: color_loss(
                color_coeffs0, color_gradients_c, directions
            )
        )(color_gradients.T)
        color_grad = jax.vmap(
            lambda color_gradients_c: color_loss_grad(
                color_coeffs0, color_gradients_c, directions
            )
        )(color_gradients.T)
        color_hessian = jax.vmap(
            lambda color_gradients_c: color_loss_hessian(
                color_coeffs0, color_gradients_c, directions
            )
        )(color_gradients.T)

        return (density_value, density_grad, density_hessian), (
            color_value,
            color_grad,
            color_hessian,
        )

    (
        (density_values, density_grads, density_hesss),
        (color_values, color_grads, color_hesss),
    ) = jax.vmap(process_single_offset)(offsets)

    density_hess = jnp.mean(density_hesss, axis=0)
    density_grad = jnp.mean(density_grads, axis=0)
    density_value = jnp.mean(density_values, axis=0)
    color_hess = jnp.mean(color_hesss, axis=0)
    color_grad = jnp.mean(color_grads, axis=0)
    color_value = jnp.mean(color_values, axis=0)

    fitted_density_coeffs = density_coeffs0 - lstsq(density_hess, density_grad)
    fitted_color_coeffs = jax.vmap(
        lambda hess_c, grad_c: color_coeffs0 - lstsq(hess_c, grad_c)
    )(color_hess, color_grad)

    if density_trilinear_interp:
        unscaled_approx_error_density = (
            0.5 * fitted_density_coeffs.T @ density_hess @ fitted_density_coeffs
            + fitted_density_coeffs @ density_grad
            + density_value
        )
    else:
        unscaled_approx_error_density = (
            0.5 * fitted_density_coeffs * density_hess * fitted_density_coeffs
            + fitted_density_coeffs * density_grad
            + density_value
        )
    unscaled_approx_error_color = jnp.sum(
        jax.vmap(
            lambda fitted_color_coeffs_c, color_hess_c, color_grad_c, color_value_c: 0.5
            * fitted_color_coeffs_c.T
            @ color_hess_c
            @ fitted_color_coeffs_c
            + fitted_color_coeffs_c @ color_grad_c
            + color_value_c
        )(fitted_color_coeffs, color_hess, color_grad, color_value)
    )
    unscaled_approx_error = unscaled_approx_error_density + unscaled_approx_error_color

    return (
        fitted_density_coeffs,
        fitted_color_coeffs,
        unscaled_approx_error * jnp.prod(voxel_size),
    )


def _trilinear_interp(coeffs, ti, tj, tk):
    lerp = lambda a, b, t: a + t * (b - a)

    v000 = coeffs[0]
    v001 = coeffs[1]
    v010 = coeffs[2]
    v011 = coeffs[3]
    v100 = coeffs[4]
    v101 = coeffs[5]
    v110 = coeffs[6]
    v111 = coeffs[7]

    v00 = lerp(v000, v001, tk)
    v01 = lerp(v010, v011, tk)
    v10 = lerp(v100, v101, tk)
    v11 = lerp(v110, v111, tk)

    v0 = lerp(v00, v01, tj)
    v1 = lerp(v10, v11, tj)

    return lerp(v0, v1, ti)
