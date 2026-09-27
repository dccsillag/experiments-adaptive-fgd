import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from icecream import ic
from PIL import Image as im

from src.functional_radiance.models.our_model.antialiasing import BoxFilter
from src.functional_radiance.models.our_model.model import (
    ExactIntegration,
    FunctionalRadianceModel,
    MSELoss,
)
from src.functional_radiance.models.our_model.preconditioners import (
    ConstantPreconditioner,
)
from src.functional_radiance.models.our_model.representation import (
    ProperVoxelGridRepresentation,
)
from src.functional_radiance.models.our_model.rkhs import SphericalHarmonicsRKHS
from src.functional_radiance.utils.data import load_nerfsynthetic_data, resize_images
from src.functional_radiance.utils.general import timed

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

BACKGROUND = jnp.array([1.0, 1.0, 1.0])
RESIZE_FACTOR = 1.0

train_data = resize_images(
    load_nerfsynthetic_data(
        "lego",
        "train",
        background=BACKGROUND,
        normalized_parametrization=False,
        rays_at_centers=True,
        n=None,
        skip=None,
    ),
    factor=RESIZE_FACTOR,
)
val_data = resize_images(
    load_nerfsynthetic_data(
        "lego",
        "val",
        background=BACKGROUND,
        normalized_parametrization=False,
        rays_at_centers=True,
        n=None,
        skip=None,
    ),
    factor=RESIZE_FACTOR,
)
test_data = resize_images(
    load_nerfsynthetic_data(
        "lego",
        "test",
        background=BACKGROUND,
        normalized_parametrization=False,
        rays_at_centers=True,
        n=None,
        skip=None,
    ),
    factor=RESIZE_FACTOR,
)


GRID_SIZE = 64

data = np.load("data/lego-64.npy")
data = np.permute_dims(data, (0, 2, 1))
data = data[:, ::-1, :]
grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
margin_x = (GRID_SIZE - data.shape[0]) // 2
margin_y = (GRID_SIZE - data.shape[1]) // 2
margin_z = (GRID_SIZE - data.shape[2]) // 2
grid[
    margin_x : data.shape[0] + margin_x,
    margin_y : data.shape[1] + margin_y,
    margin_z : data.shape[2] + margin_z,
] = data

densities = np.where(
    grid == 1.0,
    8.0,
    0.0,
)
colors = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 3, 1))


representation = ProperVoxelGridRepresentation(
    grid_size=GRID_SIZE,
    bounds_x=(-1.3, +1.3),
    bounds_y=(-1.3, +1.3),
    bounds_z=(-1.3, +1.3),
    initial_density=jnp.array(0.0),
    initial_color=jnp.array([0.5, 0.5, 0.5]),
    sh_deg=0,
    batch_size=32,
    n_random_directions=None,
    n_samples_per_voxel=24,
    sampling_pattern="naive",
    density_trilinear_interp=False,
)
model = FunctionalRadianceModel(
    representation=representation,
    loss=MSELoss(),
    antialiasing=BoxFilter(subbox_scale=1.0),
    learning_rate_schedule=lambda t: 1.0,
    preconditioner=ConstantPreconditioner(),
    background=BACKGROUND,
    density_regularization_schedule=lambda _: [],
    t_near=2.0,
    t_far=6.0,
    integration_mode=ExactIntegration(),
    direction_kernel=SphericalHarmonicsRKHS(deg=0),
    antialiasing_n_samples_per_pixel=256,
    clamp_density=None,
)

state = model.init()
ic(state.representation_state["density"].shape)
ic(state.representation_state["color"].shape)
state.representation_state["density"] = densities
state.representation_state["color"] = colors


render_image = jax.jit(
    lambda state, datum, rendering_rng: (
        model.render_image(
            state, datum.camera, shape=datum.pixels.shape, rng=rendering_rng
        )
    )
)
rng = jax.random.key(0)

with timed("render all train images"):
    rendering_rng, rng = jax.random.split(rng)
    rendered_images = train_data.map(
        lambda datum: render_image(state, datum, rendering_rng),
        batch_size=1,
    )
    rendered_images = jnp.clip(rendered_images, 0.0, 1.0)
    jax.block_until_ready(rendered_images)

    os.makedirs("data/nerf_synthetic/lego-voxelized64/train", exist_ok=True)
    for i in range(len(train_data)):
        train_img = rendered_images[i]
        train_img = np.array(train_img)
        out_img = im.fromarray(np.array(255.0 * np.array(train_img), dtype=np.uint8))
        out_img.save(f"data/nerf_synthetic/lego-voxelized64/train/r_{i}.png")

with timed("render all val images"):
    rendering_rng, rng = jax.random.split(rng)
    rendered_images = val_data.map(
        lambda datum: render_image(state, datum, rendering_rng),
        batch_size=1,
    )
    rendered_images = jnp.clip(rendered_images, 0.0, 1.0)
    jax.block_until_ready(rendered_images)

    os.makedirs("data/nerf_synthetic/lego-voxelized64/val", exist_ok=True)
    for i in range(len(val_data)):
        val_img = rendered_images[i]
        val_img = np.array(val_img)
        out_img = im.fromarray(np.array(255.0 * np.array(val_img), dtype=np.uint8))
        out_img.save(f"data/nerf_synthetic/lego-voxelized64/val/r_{i}.png")

with timed("render all test images"):
    rendering_rng, rng = jax.random.split(rng)
    rendered_images = test_data.map(
        lambda datum: render_image(state, datum, rendering_rng),
        batch_size=1,
    )
    rendered_images = jnp.clip(rendered_images, 0.0, 1.0)
    jax.block_until_ready(rendered_images)

    os.makedirs("data/nerf_synthetic/lego-voxelized64/test", exist_ok=True)
    for i in range(len(test_data)):
        test_img = rendered_images[i]
        test_img = np.array(test_img)
        out_img = im.fromarray(np.array(255.0 * np.array(test_img), dtype=np.uint8))
        out_img.save(f"data/nerf_synthetic/lego-voxelized64/test/r_{i}.png")

shutil.copyfile(
    "data/nerf_synthetic/lego/transforms_train.json",
    "data/nerf_synthetic/lego-voxelized64/transforms_train.json",
)
shutil.copyfile(
    "data/nerf_synthetic/lego/transforms_val.json",
    "data/nerf_synthetic/lego-voxelized64/transforms_val.json",
)
shutil.copyfile(
    "data/nerf_synthetic/lego/transforms_test.json",
    "data/nerf_synthetic/lego-voxelized64/transforms_test.json",
)
