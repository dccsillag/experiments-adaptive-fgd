from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.experimental import checkify
from jaxtyping import Array, Bool, Float, Int

from src.functional_radiance.utils.general import JaxRng


@dataclass
class Ray:
    origin: Float[Array, "d"]
    direction: Float[Array, "d"]

    def at(self, t: float | Float[Array, ""]) -> Float[Array, "d"]:
        return self.origin + t * self.direction


@jdc.pytree_dataclass
class Camera:
    pose: Float[Array, "4 4"]
    pose_det: Float[Array, ""]
    focal: float
    normalized_parametrization: jdc.Static[bool]
    rays_at_centers: jdc.Static[bool]

    def ray_at_normalized(self, uv: Float[Array, "d-1"]) -> Ray:
        u, v = uv
        u, v = v, u

        origin = self.pose[:3, -1]
        direction = jnp.array([u / self.focal, -v / self.focal, -1])
        direction = self.pose[:3, :3] @ direction

        if self.normalized_parametrization:
            origin = origin + 2 * direction
            direction = 6 * direction

        return Ray(origin=origin, direction=direction)

    def world_position_to_uv(self, x: Float[Array, "3"]) -> Float[Array, "2"]:
        # x = o(u,v) + t d(u,v)
        # o(u,v) = pose[:3,-1]
        # d(u,v) = pose[:3,:3] @ [u/focal, -v/focal, -1]
        #
        # x = pose[:3,-1] + t pose[:3,:3] @ [u/focal, -v/focal, -1]
        # x - pose[:3,-1] = t pose[:3,:3] @ [u/focal, -v/focal, -1]
        # pose[:3,:3]^(-1) @ (x - pose[:3,-1]) = t [u/focal, -v/focal, -1]
        # So:
        #   pose[:3,:3]^(-1) @ (x - pose[:3,-1]) = t [u/focal, -v/focal, -1]

        assert x.shape == (3,)

        vec = jnp.linalg.inv(self.pose[:3, :3]) @ (x - self.pose[:3, -1])
        assert vec.shape == (3,)

        if self.normalized_parametrization:
            t = -vec[2] - 2.0
        else:
            t = -vec[2]

        if self.normalized_parametrization:
            uv = vec[:2] * self.focal / (t + 2.0) * jnp.array([1, -1])
        else:
            uv = vec[:2] * self.focal / t * jnp.array([1, -1])
        return uv[::-1]

    def ij_to_uv(
        self, ij: Int[Array, "dim-1"], *, dims: Int[Array, "dim-1"]
    ) -> Float[Array, "dim-1"]:
        if self.rays_at_centers:
            return ij - 0.5 * jnp.array(dims) + 0.5
        else:
            return ij - 0.5 * jnp.array(dims)

    def uv_to_ij(
        self, uv: Float[Array, "dim-1"], *, dims: Int[Array, "dim-1"]
    ) -> Int[Array, "dim-1"]:
        if self.rays_at_centers:
            return jnp.around(uv + 0.5 * jnp.array(dims) - 0.5).astype(jnp.int32)
        else:
            return jnp.around(uv + 0.5 * jnp.array(dims)).astype(jnp.int32)

    def uv_to_ij_float(
        self, uv: Float[Array, "dim-1"], *, dims: Int[Array, "dim-1"]
    ) -> Float[Array, "dim-1"]:
        if self.rays_at_centers:
            return uv + 0.5 * jnp.array(dims) - 0.5
        else:
            return uv + 0.5 * jnp.array(dims)


@jdc.pytree_dataclass
class CameraAndImage:
    camera: Camera
    pixels: Float[Array, "*hw 3"]

    def is_in_bounds(self, ij: Int[Array, "dim-1"]) -> Bool[Array, ""]:
        h, w, _ = self.pixels.shape
        return (0 <= ij[0]) & (ij[0] < h) & (0 <= ij[1]) & (ij[1] < w)

    def ray_at(self, ij: Int[Array, "dim-1"] | Float[Array, "dim-1"]) -> Ray:
        return self.camera.ray_at_normalized(self.ij_to_uv(ij))

    def ij_to_uv(self, ij: Int[Array, "dim-1"]) -> Float[Array, "dim-1"]:
        *dims, c = self.pixels.shape
        assert c == 3
        return self.camera.ij_to_uv(ij, dims=dims)

    def uv_to_ij(self, uv: Float[Array, "dim-1"]) -> Int[Array, "dim-1"]:
        *dims, c = self.pixels.shape
        assert c == 3
        return self.camera.uv_to_ij(uv, dims=dims)

    def uv_to_ij_float(self, uv: Float[Array, "dim-1"]) -> Float[Array, "dim-1"]:
        *dims, c = self.pixels.shape
        assert c == 3
        return self.camera.uv_to_ij_float(uv, dims=dims)

    def world_position_to_ij(self, x: Float[Array, "dim"]) -> Int[Array, "dim-1"]:
        return self.uv_to_ij(self.camera.world_position_to_uv(x))


@jdc.pytree_dataclass
class Dataset:
    images: Float[Array, "n h w 3"]
    camera_poses: Float[Array, "n 4 4"]
    camera_pose_dets: Float[Array, "n"]
    camera_focals: Float[Array, "n"]
    camera_normalized_parametrization: jdc.Static[bool]
    rays_at_centers: jdc.Static[bool]

    @classmethod
    def from_list(
        cls,
        pairs: list[CameraAndImage],
        *,
        normalized_parametrization: bool,
        rays_at_centers: bool,
    ):
        assert all(
            datum.camera.normalized_parametrization == normalized_parametrization
            for datum in pairs
        )
        return cls(
            images=jnp.stack([datum.pixels for datum in pairs]),
            camera_poses=jnp.stack([datum.camera.pose for datum in pairs]),
            camera_pose_dets=jnp.stack([datum.camera.pose_det for datum in pairs]),
            camera_focals=jnp.stack([datum.camera.focal for datum in pairs]),
            camera_normalized_parametrization=normalized_parametrization,
            rays_at_centers=rays_at_centers,
        )

    def __len__(self) -> int:
        n = len(self.camera_focals)
        assert self.images.shape[0] == n
        assert self.camera_poses.shape[0] == n
        return n

    def __getitem__(self, i: int) -> CameraAndImage:
        return CameraAndImage(
            camera=Camera(
                pose=self.camera_poses[i],
                pose_det=self.camera_pose_dets[i],
                focal=self.camera_focals[i],
                normalized_parametrization=self.camera_normalized_parametrization,
                rays_at_centers=self.rays_at_centers,
            ),
            pixels=self.images[i],
        )

    def shuffle(self, *, rng) -> Self:
        permutation = jax.random.permutation(rng, len(self))
        return Dataset(
            images=self.images[permutation],
            camera_poses=self.camera_poses[permutation],
            camera_pose_dets=self.camera_pose_dets[permutation],
            camera_focals=self.camera_focals[permutation],
            camera_normalized_parametrization=self.camera_normalized_parametrization,
            rays_at_centers=self.rays_at_centers,
        ), permutation

    def map(self, op, *, batch_size):
        return jax.lax.map(
            lambda args: op(
                CameraAndImage(
                    camera=Camera(
                        pose=args[1],
                        pose_det=args[2],
                        focal=args[3],
                        normalized_parametrization=self.camera_normalized_parametrization,
                        rays_at_centers=self.rays_at_centers,
                    ),
                    pixels=args[0],
                )
            ),
            (self.images, self.camera_poses, self.camera_pose_dets, self.camera_focals),
            batch_size=batch_size,
        )

    def map_enumerate(self, op, *, batch_size):
        return jax.lax.map(
            lambda args: op(
                args[4],
                CameraAndImage(
                    camera=Camera(
                        pose=args[1],
                        pose_det=args[2],
                        focal=args[3],
                        normalized_parametrization=self.camera_normalized_parametrization,
                        rays_at_centers=self.rays_at_centers,
                    ),
                    pixels=args[0],
                ),
            ),
            (
                self.images,
                self.camera_poses,
                self.camera_pose_dets,
                self.camera_focals,
                jnp.arange(len(self)),
            ),
            batch_size=batch_size,
        )

    def iter_batches(self, batch_size):
        n = len(self)
        for i in range(0, n, batch_size):
            yield (
                slice(i, i + batch_size),
                Dataset(
                    images=self.images[i : i + batch_size],
                    camera_poses=self.camera_poses[i : i + batch_size],
                    camera_pose_dets=self.camera_pose_dets[i : i + batch_size],
                    camera_focals=self.camera_focals[i : i + batch_size],
                    camera_normalized_parametrization=self.camera_normalized_parametrization,
                    rays_at_centers=self.rays_at_centers,
                ),
            )
