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


class PointSampler(Protocol):
    def get_hparams_json(self) -> dict:
        pass

    def propose_points_and_directionss(
        self,
        data: Dataset,
        representation: Representation,
        representation_state: PyTree,
        *,
        rng: JaxRng,
    ) -> tuple[Float[Array, "n d"], Float[Array, "n d-1"]]:
        pass


@dataclass
class UniformRandomPointSampler(PointSampler):
    bounds_x: tuple[float, float]
    bounds_y: tuple[float, float]
    bounds_z: tuple[float, float]
    n_samples: int

    def get_hparams_json(self) -> dict:
        return {
            "point_sampler": "uniform_random",
            "bounds_x": repr(self.bounds_x),
            "bounds_y": repr(self.bounds_y),
            "bounds_z": repr(self.bounds_z),
            "n_samples": self.n_samples,
        }

    def propose_points_and_directionss(
        self,
        data: Dataset,
        representation: Representation,
        representation_state: PyTree,
        *,
        rng: JaxRng,
    ) -> tuple[Float[Array, "n d"], Float[Array, "n k d-1"]]:
        rng_x, rng_y, rng_z, rng_omega = jax.random.split(rng, 4)

        points = jnp.stack(
            (
                jax.random.uniform(
                    rng_x,
                    (self.n_samples,),
                    minval=self.bounds_x[0],
                    maxval=self.bounds_x[1],
                ),
                jax.random.uniform(
                    rng_y,
                    (self.n_samples,),
                    minval=self.bounds_y[0],
                    maxval=self.bounds_y[1],
                ),
                jax.random.uniform(
                    rng_z,
                    (self.n_samples,),
                    minval=self.bounds_z[0],
                    maxval=self.bounds_z[1],
                ),
            ),
            axis=1,
        )

        directions = jax.random.normal(rng_omega, (self.n_samples, 3))
        directions = directions / jnp.linalg.norm(directions, axis=1)[:, None]

        return points, directions[:, None, :]


@dataclass
class UniformRegularPointSampler(PointSampler):
    bounds_x: tuple[float, float]
    bounds_y: tuple[float, float]
    bounds_z: tuple[float, float]
    random_directions: bool
    n_random_direction_samples: int
    n_points_per_voxel: int
    resolution: int
    perturb_position: bool

    def get_hparams_json(self) -> dict:
        return {
            "point_sampler": "uniform_regular",
            "bounds_x": repr(self.bounds_x),
            "bounds_y": repr(self.bounds_y),
            "bounds_z": repr(self.bounds_z),
            "resolution": self.resolution,
            "random_directions": self.random_directions,
            "n_points_per_voxel": self.n_points_per_voxel,
            "n_random_direction_samples": self.n_random_direction_samples,
            "perturb_position": self.perturb_position,
        }

    def propose_points_and_directionss(
        self,
        data: Dataset,
        representation: Representation,
        representation_state: PyTree,
        *,
        rng: JaxRng,
    ) -> tuple[Float[Array, "n d"], Float[Array, "n k d-1"]]:
        if self.perturb_position:
            xx, yy, zz = jnp.meshgrid(
                jnp.linspace(*self.bounds_x, self.resolution)[:-1],
                jnp.linspace(*self.bounds_y, self.resolution)[:-1],
                jnp.linspace(*self.bounds_z, self.resolution)[:-1],
            )
            points = jnp.stack((jnp.ravel(xx), jnp.ravel(yy), jnp.ravel(zz)), axis=1)
            points = jnp.repeat(points, self.n_points_per_voxel, axis=0)

            this_rng, rng = jax.random.split(rng)
            points = points + jnp.stack(
                (
                    jax.random.uniform(
                        this_rng,
                        points.shape[0],
                        minval=0.0,
                        maxval=(self.bounds_x[1] - self.bounds_x[0])
                        / (self.resolution - 1),
                    ),
                    jax.random.uniform(
                        this_rng,
                        points.shape[0],
                        minval=0.0,
                        maxval=(self.bounds_y[1] - self.bounds_y[0])
                        / (self.resolution - 1),
                    ),
                    jax.random.uniform(
                        this_rng,
                        points.shape[0],
                        minval=0.0,
                        maxval=(self.bounds_z[1] - self.bounds_z[0])
                        / (self.resolution - 1),
                    ),
                ),
                axis=1,
            )
        else:
            xx, yy, zz = jnp.meshgrid(
                jnp.linspace(*self.bounds_x, self.resolution),
                jnp.linspace(*self.bounds_y, self.resolution),
                jnp.linspace(*self.bounds_z, self.resolution),
            )
            points = jnp.stack((jnp.ravel(xx), jnp.ravel(yy), jnp.ravel(zz)), axis=1)
            points = jnp.repeat(points, self.n_points_per_voxel, axis=0)

        if self.random_directions:
            directions = jax.random.normal(
                rng, (points.shape[0], self.n_random_direction_samples, 3)
            )
            directions = directions / jnp.linalg.norm(directions, axis=2)[:, :, None]
            return points, directions
        else:

            def direction_from_point_and_camera(point, camera):
                uv = camera.world_position_to_uv(point)
                ray = camera.ray_at_normalized(uv)
                return ray.direction

            directions = jax.vmap(
                lambda point: jnp.array(
                    [
                        direction_from_point_and_camera(point, data[i].camera)
                        for i in range(len(data))
                    ]
                )
            )(points)

            return points, directions


@dataclass
class BruteForcePointSampler(PointSampler):
    randomize_proposals: bool
    n_points_to_propose_per_ray: int
    t_near: float
    t_far: float

    def get_hparams_json(self) -> dict:
        return {
            "point_sampler": "brute_force",
            "randomize_proposals": self.randomize_proposals,
            "n_points_to_propose_per_ray": self.n_points_to_propose_per_ray,
            "t_near": self.t_near,
            "t_far": self.t_far,
        }

    def propose_points_and_directionss(
        self,
        data: Dataset,
        representation: Representation,
        representation_state: PyTree,
        *,
        rng: JaxRng,
    ) -> tuple[Float[Array, "n d"], Float[Array, "n k d-1"]]:
        if self.randomize_proposals:
            this_rng, rng = jax.random.split(rng)
            ts = jax.random.uniform(
                this_rng,
                shape=(self.n_points_to_propose_per_ray,),
                minval=self.t_near,
                maxval=self.t_far,
            )
        else:
            ts = jnp.linspace(self.t_near, self.t_far, self.n_points_to_propose_per_ray)

        def generate_points(
            ij: Int[Array, "2"],
            t: Float[Array, ""],
            perturbation: Float[Array, "2"],
        ) -> Float[Array, "m d"]:
            ray = datum.ray_at(ij + perturbation)
            point = ray.at(t)
            viewing_direction = point - ray.origin
            return jnp.concatenate((point, viewing_direction), axis=-1)

        pointss = []
        directionss = []
        for i in tqdm(range(len(data))):
            datum = data[i]
            this_rng, rng = jax.random.split(rng)
            h, w, _ = datum.pixels.shape

            perturbations_ijt = jax.random.uniform(
                this_rng, shape=(h, w, len(ts), 2), minval=-0.5, maxval=+0.5
            )

            points_and_directions = jax.vmap(
                lambda i, perturbations_jt: jax.vmap(
                    lambda j, perturbations_t: jax.vmap(
                        lambda t, perturbation: generate_points(
                            jnp.array((i, j)), t, perturbation
                        )
                    )(ts, perturbations_t)
                )(jnp.arange(w), perturbations_jt)
            )(jnp.arange(h), perturbations_ijt)
            points_and_directions = points_and_directions.reshape(
                (-1, points_and_directions.shape[-1])
            )
            assert points_and_directions.shape[-1] == 6

            pointss.append(points_and_directions[:, :3])
            directionss.append(points_and_directions[:, 3:])

        return (
            jnp.concatenate(pointss, axis=0),
            jnp.concatenate(directionss, axis=0)[:, None, :],
        )
