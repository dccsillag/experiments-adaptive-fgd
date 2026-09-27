import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Self

import dill
import jax
import jax.numpy as jnp
from icecream import ic
from jax.experimental import checkify
from jaxtyping import Array, Float, Int, PyTree
from tqdm import tqdm

from src.functional_radiance.models.our_model.antialiasing import Antialiasing
from src.functional_radiance.models.our_model.preconditioners import Preconditioner
from src.functional_radiance.models.our_model.regularization import (
    DensityRegularization,
)
from src.functional_radiance.models.our_model.representation import Representation
from src.functional_radiance.models.our_model.rkhs import RKHS
from src.functional_radiance.utils.general import (
    BaseLearner,
    JaxRng,
    cumulative_trapezoid,
    jax_predict,
    repr_lambda,
    viewing_direction_to_unit_circle,
)
from src.functional_radiance.utils.models import Camera, CameraAndImage, Dataset, Ray


@dataclass(frozen=True)
class RiemannSum:
    n_samples: int


@dataclass(frozen=True)
class TrapezoidalSum:
    n_samples: int


@dataclass(frozen=True)
class NerfSum:
    n_samples: int
    use_exp: bool


@dataclass(frozen=True)
class ExactIntegration:
    pass


@jax.tree_util.register_dataclass
@dataclass
class FunctionalRadianceModelState:
    representation_state: PyTree
    prev_representation_state: PyTree
    epoch_num: int


class Loss(Protocol):
    def get_hparams_json(self) -> dict:
        pass

    def __call__(self, y_diff: Float[Array, "3"]) -> Float[Array, ""]:
        pass


@dataclass
class MSELoss(Loss):
    def get_hparams_json(self) -> dict:
        return {"loss": "mse"}

    def __call__(self, y_diff: Float[Array, "3"]) -> Float[Array, ""]:
        return 0.5 * jnp.sum(y_diff**2)


@dataclass
class MAELoss(Loss):
    def get_hparams_json(self) -> dict:
        return {"loss": "mae"}

    def __call__(self, y_diff: Float[Array, "3"]) -> Float[Array, ""]:
        @jax.custom_jvp
        def f(d):
            return jnp.sum(jnp.abs(d))

        @f.defjvp
        def f_jvp(primals, tangents):
            (x,), (x_dot,) = primals, tangents
            primal_out = f(x)
            tangent_out = jnp.sum(jnp.sign(x) * x_dot)
            return primal_out, tangent_out

        return f(y_diff)


@dataclass
class ElementwiseHuberLoss(Loss):
    delta: float

    def get_hparams_json(self) -> dict:
        return {"loss": "huber", "delta": self.delta}

    def __call__(self, y_diff: Float[Array, "3"]) -> Float[Array, ""]:
        return (
            jnp.sum(
                jnp.where(
                    jnp.abs(y_diff) <= self.delta,
                    0.5 * y_diff**2,
                    self.delta * (jnp.abs(y_diff) - 0.5 * self.delta),
                )
            )
            / self.delta
        )


@dataclass
class FunctionalRadianceModel:
    representation: Representation
    loss: Loss
    antialiasing: Antialiasing
    learning_rate_schedule_density: Callable[[int], float]
    learning_rate_schedule_color: Callable[[int], float]
    preconditioner: Preconditioner
    background: Float[Array, "3"]
    density_regularization_schedule: Callable[[int], list[DensityRegularization]]
    t_near: float
    t_far: float
    integration_mode: RiemannSum | TrapezoidalSum | ExactIntegration
    direction_kernel: RKHS
    antialiasing_n_samples_per_pixel: int
    clamp_density: tuple[float, float] | None

    def get_hparams_json(self) -> dict:
        return {
            "representation": self.representation.get_hparams_json(),
            "loss": self.loss.get_hparams_json(),
            "antialiasing": self.antialiasing.get_hparams_json(),
            "learning_rate_schedule_density": repr_lambda(
                self.learning_rate_schedule_density
            ),
            "learning_rate_schedule_color": repr_lambda(
                self.learning_rate_schedule_color
            ),
            "preconditioner": self.preconditioner.get_hparams_json(),
            "background": repr(self.background),
            "density_regularization_schedule": repr_lambda(
                self.density_regularization_schedule
            ),
            "t_near": self.t_near,
            "t_far": self.t_far,
            "integration_mode": repr(self.integration_mode),
            "direction_kernel": repr(self.direction_kernel),
            "antialiasing_n_samples_per_pixel": self.antialiasing_n_samples_per_pixel,
            "clamp_density": self.clamp_density,
        }

    def init(self) -> FunctionalRadianceModelState:
        return FunctionalRadianceModelState(
            representation_state=self.representation.init(),
            prev_representation_state=self.representation.init(),
            epoch_num=0,
        )

    def density_function(self, state, x):
        return self.representation.eval_density(state.representation_state, x)

    def color_function(self, state, x, direction):
        return self.representation.eval_color(state.representation_state, x, direction)

    def _render_within_interval(
        self,
        state: FunctionalRadianceModelState,
        *,
        ray: Ray,
        t_start: float,
        t_end: float,
    ) -> tuple[
        Float[Array, "3"],  # rendered color
        Float[Array, ""],  # transmittance at t_end
    ]:
        # NOTE: no background here!! This is by design.
        match self.integration_mode:
            case ExactIntegration():
                return self.representation.render_within_interval(
                    state.representation_state, ray=ray, t_start=t_start, t_end=t_end
                )

            case RiemannSum(n_samples):
                ts = (  # `n_samples` points between 0 and 1, ending at 1 but skipping 0
                    jnp.linspace(t_start, t_end, n_samples + 1)[1:]
                )
                step_size = (t_end - t_start) / n_samples
                densities_along_ray = jax.vmap(
                    lambda t: self.density_function(state, ray.at(t))
                )(ts)
                colors_along_ray = jax.vmap(
                    lambda t: self.color_function(state, ray.at(t), ray.direction)
                )(ts)
                transmittances_along_ray = jnp.exp(
                    -(jnp.cumsum(densities_along_ray) * step_size)
                )
                rendered_colors_along_ray = (
                    jnp.cumsum(
                        transmittances_along_ray[:, None]
                        * densities_along_ray[:, None]
                        * colors_along_ray,
                        axis=0,
                    )
                    * step_size
                )
                return (
                    rendered_colors_along_ray[-1],
                    transmittances_along_ray[-1],
                )

            case TrapezoidalSum(n_samples):
                density_at_start = self.density_function(state, ray.at(t_start))
                color_at_start = self.color_function(
                    state, ray.at(t_start), ray.direction
                )
                step_size = (t_end - t_start) / (n_samples - 1)

                def rendering_loop(k, carry):
                    (prev_density, prev_value, rendered_color, density_integral) = carry

                    t = t_start + k * step_size
                    density = self.density_function(state, ray.at(t))
                    color = self.color_function(state, ray.at(t), ray.direction)

                    density_integral = (
                        density_integral + (prev_density + density) * 0.5 * step_size
                    )
                    transmittance = jnp.exp(-density_integral)
                    value = transmittance * density * color
                    rendered_color = (
                        rendered_color + (prev_value + value) * 0.5 * step_size
                    )

                    prev_density = density
                    prev_value = value

                    return prev_density, prev_value, rendered_color, density_integral

                (
                    _prev_density,
                    _prev_value,
                    rendered_color,
                    density_integral,
                ) = jax.lax.fori_loop(
                    1,
                    n_samples,
                    rendering_loop,
                    (
                        density_at_start,  # prev_density
                        1 * density_at_start * color_at_start,  # prev_value
                        jnp.array([0.0, 0.0, 0.0]),  # rendered_color
                        0.0,  # density_integral
                    ),
                )

                transmittance = jnp.exp(-density_integral)
                return rendered_color, transmittance

            case NerfSum(n_samples, use_exp=True):
                density_at_start = self.density_function(state, ray.at(t_start))
                color_at_start = self.color_function(
                    state, ray.at(t_start), ray.direction
                )
                step_size = (t_end - t_start) / (n_samples - 1)

                def rendering_loop(k, carry):
                    (rendered_color, density_integral) = carry

                    t = t_start + k * step_size
                    density = self.density_function(state, ray.at(t))
                    color = self.color_function(state, ray.at(t), ray.direction)

                    rendered_color = (
                        rendered_color
                        + jnp.exp(-density_integral)
                        * (1 - jnp.exp(-density * step_size))
                        * color
                    )
                    density_integral = density_integral + density * step_size

                    return rendered_color, density_integral

                (
                    rendered_color,
                    density_integral,
                ) = jax.lax.fori_loop(
                    0,
                    n_samples,
                    rendering_loop,
                    (
                        jnp.array([0.0, 0.0, 0.0]),  # rendered_color
                        0.0,  # density_integral
                    ),
                )

                transmittance = jnp.exp(-density_integral)
                return rendered_color, transmittance

    def _render_at_t(self, state: FunctionalRadianceModelState, *, ray: Ray, t: float):
        (
            rendered_color_at_t,
            transmittance_at_t,
        ) = self._render_within_interval(state, ray=ray, t_start=self.t_near, t_end=t)
        (
            rendered_color_at_far,
            transmittance_at_far,
        ) = self._render_within_interval(
            state, ray=ray, t_start=self.t_near, t_end=self.t_far
        )
        return (
            # full render:
            rendered_color_at_far + transmittance_at_far * self.background,
            # render up to k:
            rendered_color_at_t,
            # transmittance at k:
            transmittance_at_t,
        )

    def render_ray(
        self, state: FunctionalRadianceModelState, ray: Ray
    ) -> Float[Array, "3"]:
        (
            rendered_color_at_far,
            transmittance_at_far,
        ) = self._render_within_interval(
            state, ray=ray, t_start=self.t_near, t_end=self.t_far
        )
        return rendered_color_at_far + transmittance_at_far * self.background

    def render_image(
        self,
        state: FunctionalRadianceModelState,
        camera: Camera,
        *,
        shape: tuple[int, ...],
        rng: JaxRng,
    ) -> Float[Array, "*hw 3"]:
        h, w, _ = shape

        perturbations_ijk, weights_ijk = self.antialiasing.sample_for_rendering(
            shape=(h, w, self.antialiasing_n_samples_per_pixel),
            rng=rng,
        )
        iii, jjj = jnp.meshgrid(jnp.arange(h), jnp.arange(w))
        iii, jjj = iii.T, jjj.T
        return jax.vmap(
            lambda ii, jj, perturbations_ik, weights_ik: jax.vmap(
                lambda i, j, perturbations_k, weights_k: jnp.mean(
                    jax.vmap(
                        lambda perturbation, weight: self.render_ray(
                            state,
                            camera.ray_at_normalized(
                                camera.ij_to_uv(
                                    jnp.array((i, j)) + perturbation,
                                    dims=jnp.array((h, w)),
                                )
                            ),
                        )
                        * weight
                    )(perturbations_k, weights_k),
                    axis=0,
                )
            )(ii, jj, perturbations_ik, weights_ik),
        )(iii, jjj, perturbations_ijk, weights_ijk)

    def apply_updates(
        self,
        state: FunctionalRadianceModelState,
        gradient_approx: PyTree,
    ) -> FunctionalRadianceModelState:
        # update the representation
        learning_rate_density = self.learning_rate_schedule_density(state.epoch_num)
        learning_rate_color = self.learning_rate_schedule_color(state.epoch_num)
        new_representation_state = self.representation.add(
            state.representation_state,
            self.representation.scale_color(
                -learning_rate_color,
                self.representation.scale_density(
                    -learning_rate_density, gradient_approx
                ),
            ),
            assume_compatible=True,
        )
        if self.clamp_density is not None:
            new_representation_state = self.representation.clamp_density(
                new_representation_state, self.clamp_density
            )

        return FunctionalRadianceModelState(
            representation_state=new_representation_state,
            prev_representation_state=state.prev_representation_state,
            epoch_num=state.epoch_num + 1,
        )

    def momentum_step(
        self, state: FunctionalRadianceModelState
    ) -> FunctionalRadianceModelState:
        # v_(k+1) = x_k + c (x_k - x_(k-1))
        #         = (1 + c) x_k - c x_(k-1)
        # where c = ((k-1)/(k+2)).
        k = state.epoch_num + 1
        c = (k - 2) / (k + 1)

        new_representation_state = self.representation.add(
            self.representation.scale(1 + c, state.representation_state),
            self.representation.scale(-c, state.prev_representation_state),
            assume_compatible=True,
        )

        return FunctionalRadianceModelState(
            representation_state=new_representation_state,
            prev_representation_state=state.representation_state,
            epoch_num=state.epoch_num,
        )

    def functional_gradient(
        self,
        state: FunctionalRadianceModelState,
        x: Float[Array, "d"],
        viewing_directions: Float[Array, "k d"],
        *,
        data: Dataset,
        full_data: Dataset,
        rendered_images: list[Float[Array, "h w 3"]],
    ) -> tuple[Float[Array, ""], Float[Array, "k 3"]]:
        out_densities, out_colors = data.map_enumerate(
            lambda i, datum: self._functional_gradient_for_datum(
                state,
                x,
                viewing_directions,
                datum=datum,
                rendered_image=rendered_images[i],
            ),
            batch_size=None,
        )

        density_gradient = jnp.mean(out_densities, axis=0)
        for regularization in self.density_regularization_schedule(state.epoch_num):
            density_gradient = density_gradient + regularization.grad_over_space(
                x, density_function=lambda x: self.density_function(state, x)
            )

        color_gradient = jnp.mean(out_colors, axis=0)

        preconditioner_val = self.preconditioner(x, full_data=full_data)
        return (
            preconditioner_val * density_gradient,
            preconditioner_val * color_gradient,
        )

    def _functional_gradient_for_datum(
        self,
        state: FunctionalRadianceModelState,
        x: Float[Array, "d"],
        viewing_directions: Float[Array, "k d"],
        *,
        datum: CameraAndImage,
        rendered_image: Float[Array, "h w 3"],
    ) -> tuple[Float[Array, ""], Float[Array, "k 3"]]:
        """Compute the functional gradient w.r.t. the density and color with the current parameters, evaluated at the given point `x` and viewing direction `viewing_direction`."""

        # Find the matching pixel and ray in image
        matching_uv = datum.camera.world_position_to_uv(x)
        matching_ray = datum.camera.ray_at_normalized(matching_uv)
        matching_ij_float = datum.uv_to_ij_float(matching_uv)
        matching_ij = datum.uv_to_ij(matching_uv)

        color_diff = jnp.zeros(3)
        for delta_i in range(
            0 - self.antialiasing.support_radius,
            0 + self.antialiasing.support_radius + 1,
        ):
            for delta_j in range(
                0 - self.antialiasing.support_radius,
                0 + self.antialiasing.support_radius + 1,
            ):
                delta = jnp.array([delta_i, delta_j])
                matching_pixel = datum.pixels[*(matching_ij + delta)]
                matching_render = rendered_image[*(matching_ij + delta)]
                weight = self.antialiasing.kernel_function(
                    matching_ij_float - (matching_ij + delta)
                )
                color_diff = color_diff + weight * jax.grad(self.loss)(
                    matching_render - matching_pixel
                )

        # Get how far along the matching ray we get the point `x`, and compute the transmittance.
        matching_t = ((x - matching_ray.origin) @ matching_ray.direction) / (
            matching_ray.direction @ matching_ray.direction
        )

        (
            rendered_full,
            rendered_up_to_t,
            transmittance_at_t,
        ) = self._render_at_t(state, ray=matching_ray, t=matching_t)
        matching_density = self.density_function(state, x)
        matching_color = self.color_function(state, x, matching_ray.direction)

        density_gradient = color_diff @ (
            transmittance_at_t * matching_color - (rendered_full - rendered_up_to_t)
        )
        for regularization in self.density_regularization_schedule(state.epoch_num):
            density_gradient = density_gradient + regularization.grad_over_ray(
                t=matching_t,
                rendered_full=rendered_full,
                rendered_up_to_t=rendered_up_to_t,
                transmittance_at_t=transmittance_at_t,
                density_at_t=matching_density,
                color_at_t=matching_color,
            )

        color_kernels = jax.vmap(
            lambda viewing_direction: self.direction_kernel.kernel(
                matching_ray.direction, viewing_direction
            )
        )(viewing_directions)
        color_gradient = (
            color_diff[None, :]
            * transmittance_at_t
            * matching_density
            * color_kernels[:, None]
        )

        a_i = (
            (datum.camera.focal**2)
            * ((self.t_near <= matching_t) & (matching_t <= self.t_far))
        ) / (
            datum.pixels.shape[0]
            * datum.pixels.shape[1]
            * datum.camera.pose_det
            * matching_t**2
        )

        return a_i * density_gradient, a_i * color_gradient
