import argparse
import getpass
import json
import os
import socket
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pickle
from functools import partial
from math import ceil, cos, floor, pi

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from icecream import ic
from jax import config
from lpips import LPIPS
from PIL import Image as im
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from src.functional_radiance.models.our_model.antialiasing import (
    Antialiasing,
    BoxFilter,
    BsplineFilter,
)
from src.functional_radiance.models.our_model.model import (
    ElementwiseHuberLoss,
    ExactIntegration,
    FunctionalRadianceModel,
    MAELoss,
    MSELoss,
    NerfSum,
    RiemannSum,
    TrapezoidalSum,
)
from src.functional_radiance.models.our_model.point_sampling import (
    BruteForcePointSampler,
    UniformRandomPointSampler,
    UniformRegularPointSampler,
)
from src.functional_radiance.models.our_model.preconditioners import (
    ConstantPreconditioner,
    ForegroundPreconditioner,
    Preconditioner,
    SupervisionPreconditioner,
)
from src.functional_radiance.models.our_model.regularization import (
    LpRegularization,
    LpRegularizationAlongRay,
    PlenoctreeSparsityRegularization,
)
from src.functional_radiance.models.our_model.representation import (
    ProperVoxelGridRepresentation,
    Representation,
)
from src.functional_radiance.models.our_model.rkhs import (
    ConstantRKHS,
    SphericalHarmonicsRKHS,
    ZonalRKHS,
)
from src.functional_radiance.utils.data import load_nerfsynthetic_data, resize_images
from src.functional_radiance.utils.eval import get_nerf_prediction_folder, get_psnr
from src.functional_radiance.utils.general import atomic_write, timed
from src.functional_radiance.utils.log import get_readable_time

matplotlib.use("Agg")

# config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--scene",
    type=str,
    required=True,
    help="which scene to train on",
)

parser.add_argument(
    "--setting",
    type=str,
    choices=["full", "small", "paper", "medium", "small_100imgs"],
    required=True,
    help="which size of the scene to use",
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=False,
    help="where to write the outputs; by default, generates a unique directory in the models/ dir",
)
parser.add_argument(
    "--test_skip",
    default=8,
)
parser.add_argument(
    "--val_skip",
    default=8,
)
parser.add_argument(
    "--evaluate_every",
    type=int,
    help="How many epochs/steps between evaluation rounds",
)

parser.add_argument(
    "-r",
    "--representation",
    default="proper_voxel_grid",
    choices=["proper_voxel_grid"],
    help="which representation to use",
)

parser.add_argument(
    "--grid_size",
    default=60,
    type=int,
    help="if voxel_grid representation is used, determines grid size",
)
parser.add_argument(
    "--fixed_grid", action="store_true", help="if used, do not adapt the grid"
)
parser.add_argument(
    "--grid_radius",
    # default=1.5,
    default=1.3,
    type=float,
    help="if voxel_grid representation is used, determines the size of the cube that contains the grid",
)

parser.add_argument(
    "--trilinear_interp",
    action="store_true",
    help="use trilinnear interpolation in the grid representation",
)

parser.add_argument(
    "--sh_deg", default=1, type=int, help="degree of spherical harmonics"
)

parser.add_argument(
    "-nr",
    "--normalize_rays",
    action="store_true",
    help="if activated, parametrize rays in [0, 1]. Otherwise, [2, 6] as in NeRF",
)
parser.add_argument(
    "-nrc",
    "--no_rays_at_centers",
    dest="rays_at_centers",
    action="store_false",
    help="if activated, rays are cast from the corners of pixels, rather than centers. This corresponds to a bug present in the original NeRF code",
)
parser.add_argument(
    "-e",
    "--max_epochs",
    type=int,
    default=200,
)

parser.add_argument(
    "--batch_size_render",
    type=int,
    default=1,
)
parser.add_argument(
    "--batch_size_gradient",
    type=int,
    default=8,
)

parser.add_argument(
    "--sgd",
    action="store_true",
    help="Optimize with mini-batch SGD (with batch size given by --batch_size_gradient), rather than full-batch GD",
)


parser.add_argument(
    "--lr_density", help="(default: %(default)s)", type=float, default=30
)
parser.add_argument("--lr_color", help="(default: %(default)s)", type=float, default=30)

parser.add_argument(
    "--n_random_direction_samples",
    help="number random directions for approximation of gradient wrt color. If not passed, uses the directions connecting to the training cameras",
    type=int,
)

parser.add_argument("--momentum", action="store_true", help="use Nesterov momentum")

parser.add_argument(
    "--n_samples_for_antialiasing",
    help="(default: %(default)s)",
    type=int,
    default=2048,
)

parser.add_argument(
    "--min_density",
    type=float,
    default=0.0,
    help="Minimum density for clamping; if none is passed, don't clamp",
)
parser.add_argument(
    "--max_density",
    type=float,
    default=10.0,
    help="Maximum density for clamping; if none is passed, don't clamp",
)

parser.add_argument(
    "--antialiasing_subbox_scale",
    type=float,
    default=1.0,
    help="% of pixel size to use for the box kernel for antialiasing (1.0 = full pixel)",
)

parser.add_argument(
    "--density_mask_epoch",
    type=int,
    help="Epoch at which to freeze which voxels to update",
)
parser.add_argument(
    "--density_mask_threshold",
    type=float,
    help="At the `--density_masak_epoch` epoch, voxels with densities below this threhsold will no longer be updated.",
)

parser.add_argument(
    "--added_rendering_gaussian_noise",
    type=float,
    default=0.0,
    help="amount of added Gaussian noise to the rendering during training",
)

args = parser.parse_args()


match args.setting:
    case "full":
        resize_factor = 1.0
        n_train = None  # uses all images available
    case "small":
        resize_factor = 0.1
        n_train = 24
    case "small_100imgs":
        resize_factor = 0.1
        n_train = 100
    case "medium":
        resize_factor = 1.0
        n_train = 24
    case "paper":
        resize_factor = 0.2
        n_train = 24

BACKGROUND = jnp.array([1.0, 1.0, 1.0])


# Load data:
train_data = resize_images(
    load_nerfsynthetic_data(
        args.scene,
        "train",
        background=BACKGROUND,
        normalized_parametrization=args.normalize_rays,
        rays_at_centers=args.rays_at_centers,
        n=n_train,
        skip=None,
    ),
    factor=resize_factor,
)
val_data = resize_images(
    load_nerfsynthetic_data(
        args.scene,
        "val",
        background=BACKGROUND,
        normalized_parametrization=args.normalize_rays,
        rays_at_centers=args.rays_at_centers,
        n=None,
        skip=args.val_skip,
    ),
    factor=resize_factor,
)
test_data = resize_images(
    load_nerfsynthetic_data(
        args.scene,
        "test",
        background=BACKGROUND,
        normalized_parametrization=args.normalize_rays,
        rays_at_centers=args.rays_at_centers,
        n=None,
        skip=args.test_skip,
    ),
    factor=resize_factor,
)

val_data, test_data = test_data, val_data  # HACK so that we eval on test

print(f"Image sizes: {train_data[0].pixels.shape = }")

if args.normalize_rays:
    t_near, t_far = 0.0, 1.0
else:
    t_near, t_far = 2.0, 6.0  # from NeRF

representation: Representation = {
    "proper_voxel_grid": ProperVoxelGridRepresentation(
        init_grid_size=args.grid_size,
        bounds_x=(-args.grid_radius, +args.grid_radius),
        bounds_y=(-args.grid_radius, +args.grid_radius),
        bounds_z=(-args.grid_radius, +args.grid_radius),
        initial_density=jnp.array(0.0),
        initial_color=jnp.array([0.5, 0.5, 0.5]),
        sh_deg=args.sh_deg,
        batch_size=64,
        n_random_directions=args.n_random_direction_samples,
        n_samples_per_voxel=8,
        sampling_pattern="symmetrical" if args.trilinear_interp else "naive",
        density_trilinear_interp=args.trilinear_interp,
    ),
}[args.representation]

antialiasing: Antialiasing = BoxFilter(subbox_scale=args.antialiasing_subbox_scale)
# antialiasing: Antialiasing = BsplineFilter(importance_sample=True)

# preconditioner: Preconditioner = ConstantPreconditioner()
# preconditioner: Preconditioner = SupervisionPreconditioner(t_near=t_near, t_far=t_far, pow=1.0)
# preconditioner: Preconditioner = ForegroundPreconditioner(t_near=t_near, t_far=t_far, background_lr_mult=1e-7)
preconditioner: Preconditioner = ForegroundPreconditioner(
    t_near=t_near, t_far=t_far, background_lr_mult=0.0
)

# # (2) preconditioner dump:
# ii, jj, kk = jnp.meshgrid(
#     jnp.linspace(-args.grid_radius, args.grid_radius, 200),
#     jnp.linspace(-args.grid_radius, args.grid_radius, 200),
#     jnp.linspace(-args.grid_radius, args.grid_radius, 200),
# )
# points = jnp.stack((np.ravel(ii), np.ravel(jj), np.ravel(kk)), axis=1)
# precond = jax.vmap(lambda x: preconditioner(x, full_data=train_data))(points).reshape(
#     ii.shape
# )
# np.save(f"preconditioner.npy", precond)
# breakpoint()

match args.min_density, args.max_density:
    case None, None:
        clamp_density = None
    case inf, None:
        clamp_density = inf, jnp.inf
    case None, sup:
        clamp_density = -jnp.inf, sup
    case inf, sup:
        clamp_density = inf, sup
# Setup model
model = FunctionalRadianceModel(
    representation=representation,
    antialiasing=antialiasing,
    learning_rate_schedule_density=lambda epoch_num: args.lr_density,
    learning_rate_schedule_color=lambda epoch_num: args.lr_color,
    loss=MSELoss(),
    preconditioner=preconditioner,
    integration_mode=ExactIntegration(),
    background=BACKGROUND,
    density_regularization_schedule=lambda i: [],
    t_near=t_near,
    t_far=t_far,
    direction_kernel=SphericalHarmonicsRKHS(deg=args.sh_deg),
    antialiasing_n_samples_per_pixel=args.n_samples_for_antialiasing,
    clamp_density=clamp_density,
)

hparams = model.get_hparams_json() | {
    "scene": args.scene,
    "setting": args.setting,
    "resize_factor": resize_factor,
    "n_train": n_train,
    "test_skip": args.test_skip,
    "val_skip": args.val_skip,
    "normalize_rays": args.normalize_rays,
    "rays_at_centers": args.rays_at_centers,
    "max_epochs": args.max_epochs,
    "batch_size_render": args.batch_size_render,
    "batch_size_gradient": args.batch_size_gradient,
    "all_args": vars(args),
}
print(f"{hparams = }")

if args.output_dir is not None:
    output_folder = args.output_dir
else:
    output_folder = f"models/our_model/{args.scene}-{args.setting}/test_skip_{args.test_skip}-m_{args.representation}/{get_readable_time()}"

print(f"{output_folder = }")
# Setup where we will save the outputs:
os.makedirs(output_folder)

os.makedirs(output_folder + "/predictions/train")
os.makedirs(output_folder + "/predictions/val")
os.makedirs(output_folder + "/checkpoints")
os.makedirs(output_folder + "/image_grid")

with open(f"{output_folder}/hparams.json", "w") as file:
    json.dump(hparams, file)
with open(f"{output_folder}/env.json", "w") as file:
    json.dump(
        {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "env": dict(os.environ),
        },
        file,
    )

# Training loop, together with eval at each step:
train_mses = []
train_psnrs = []
train_ssims = []
train_lpips = []
val_mses = []
val_psnrs = []
val_ssims = []
val_lpips = []
density_norms = []
color_norms = []
lpips_impl = LPIPS(net="vgg")
lpips = lambda a, b: float(
    np.squeeze(
        lpips_impl(
            torch.permute(
                torch.from_numpy(np.asarray(2 * a.astype(np.float32) - 1)), (2, 0, 1)
            )[None, ...],
            torch.permute(
                torch.from_numpy(np.asarray(2 * b.astype(np.float32) - 1)), (2, 0, 1)
            )[None, ...],
        )
        .detach()
        .numpy()
    )
)

rng = jax.random.key(0)
state = model.init()
render_image = jax.jit(
    lambda state, datum, rendering_rng: (
        model.render_image(
            state, datum.camera, shape=datum.pixels.shape, rng=rendering_rng
        )
    )
)


@jax.jit
def approximate_gradient(
    state,
    representation_state,
    full_data,
    data,
    rendered_images,
    this_rng,
):
    return model.representation.approx_gradient(
        representation_state,
        lambda x, omegas: model.functional_gradient(
            state,
            x,
            omegas,
            data=data,
            full_data=full_data,
            rendered_images=rendered_images,
        ),
        data=full_data,
        rng=this_rng,
    )


start_time = time.time()

should_subdivide = False
for epoch_num in range(args.max_epochs):
    print(f"Epoch #{epoch_num}")

    # Let's do the train step. This is done in a few parts.
    start_train = time.time()

    # 1. apply momentum, if enabled
    if args.momentum:
        with timed("momentum step"):
            state = model.momentum_step(state)
            jax.block_until_ready(state)

    # 2. render all images according to our model (this will be used in the functional gradient computations)
    with timed("render all train images"):
        rendering_rng, rng = jax.random.split(rng)
        rendered_images = train_data.map(
            lambda datum: render_image(state, datum, rendering_rng),
            batch_size=args.batch_size_render,
        )
        noise_rng, rng = jax.random.split(rng)
        rendered_images = jnp.clip(
            rendered_images
            + args.added_rendering_gaussian_noise
            * jax.random.normal(noise_rng, rendered_images.shape),
            0.0,
            1.0,
        )
        jax.block_until_ready(rendered_images)

    # 3. approximate the gradient
    EPS = 0.4
    with timed("approximating the gradient"):
        while True:
            ic(state.representation_state["density"].shape)

            if args.sgd:
                shuffle_rng, rng = jax.random.split(rng)
                shuffled_data, permutation = train_data.shuffle(rng=shuffle_rng)
                batch_slice, batch_train_data = next(
                    shuffled_data.iter_batches(args.batch_size_gradient)
                )

                approx_rng, rng = jax.random.split(rng)
                gradient_approx, approx_error = approximate_gradient(
                    state,
                    state.representation_state,
                    train_data,
                    batch_train_data,
                    rendered_images[permutation][ic(batch_slice)],
                    approx_rng,
                )
                assert not jnp.any(jnp.isnan(gradient_approx["density"]))
                assert not jnp.any(jnp.isnan(gradient_approx["color"]))
            else:
                gradient_approx = None
                approx_error = 0.0
                approx_rng, rng = jax.random.split(rng)
                for batch_slice, batch_train_data in train_data.iter_batches(
                    args.batch_size_gradient
                ):
                    acc_gradient_approx, acc_approx_error = approximate_gradient(
                        state,
                        state.representation_state,
                        train_data,
                        batch_train_data,
                        rendered_images[ic(batch_slice)],
                        approx_rng,
                    )
                    assert not jnp.any(jnp.isnan(acc_gradient_approx["density"]))
                    assert not jnp.any(jnp.isnan(acc_gradient_approx["color"]))
                    acc_gradient_approx = model.representation.scale(
                        len(batch_train_data), acc_gradient_approx
                    )
                    acc_approx_error = len(batch_train_data) * acc_approx_error

                    if gradient_approx is None:
                        gradient_approx = acc_gradient_approx
                    else:
                        gradient_approx = model.representation.add(
                            gradient_approx, acc_gradient_approx, assume_compatible=True
                        )
                    approx_error = approx_error + acc_approx_error
                gradient_approx = model.representation.scale(
                    1.0 / len(train_data), gradient_approx
                )
                approx_error = approx_error / len(train_data)

            gradient_norm = model.representation.density_norm(
                gradient_approx
            ) + model.representation.color_norm(gradient_approx)
            rel_error = jnp.sqrt(approx_error / gradient_norm)
            ic(rel_error)

            if (
                rel_error < EPS
                or state.representation_state["density"].shape[0] >= 128
                or args.fixed_grid
            ):
                break
            else:
                with timed("subdividing"):
                    state.representation_state = model.representation.subdivide(
                        state.representation_state
                    )
                    state.prev_representation_state = model.representation.subdivide(
                        state.prev_representation_state
                    )

                    # setup grid of locations
                    left_linspace = lambda a, b, grid_size: (
                        jnp.linspace(a, b, grid_size + 1)[:-1],
                        (b - a) / grid_size,
                    )
                    xs, delta_x = left_linspace(
                        *model.representation.bounds_x,
                        state.representation_state["density"].shape[0],
                    )
                    ys, delta_y = left_linspace(
                        *model.representation.bounds_y,
                        state.representation_state["density"].shape[1],
                    )
                    zs, delta_z = left_linspace(
                        *model.representation.bounds_z,
                        state.representation_state["density"].shape[2],
                    )
                    xs = xs + delta_x * 0.5
                    ys = ys + delta_y * 0.5
                    zs = zs + delta_z * 0.5

                    # apply the preconditioner after subdivision
                    mask = jax.vmap(
                        lambda x: jax.vmap(
                            lambda y: jax.vmap(
                                lambda z: preconditioner(
                                    jnp.array([x, y, z]), full_data=train_data
                                )
                            )(zs)
                        )(ys)
                    )(xs)
                    state.representation_state["density"] = jnp.where(
                        mask, state.representation_state["density"], 0.0
                    )
                    state.prev_representation_state["density"] = jnp.where(
                        mask, state.prev_representation_state["density"], 0.0
                    )

        jax.block_until_ready(gradient_approx)

    # 4. finally, update the representation and training state
    with timed("apply updates"):
        state = model.apply_updates(state, gradient_approx)
        jax.block_until_ready(state)

    if args.density_mask_epoch is not None and args.density_mask_threshold is not None:
        if epoch_num == args.density_mask_epoch:
            mask = state.representation_state["density"] >= args.density_mask_threshold
        if epoch_num >= args.density_mask_epoch:
            state.representation_state["density"] = jnp.where(
                mask, state.representation_state["density"], 0.0
            )

    jax.block_until_ready(state)
    dt = time.time() - start_train

    time_info = {"epoch": epoch_num, "delta_time": dt}

    with open(output_folder + "/times.jsonl", "a") as f:
        f.write(json.dumps(time_info) + "\n")

    # Train step is done. Now we save and do eval.
    print("done")

    if args.evaluate_every is not None and epoch_num % args.evaluate_every != 0:
        continue
    if args.evaluate_every is None and epoch_num < args.max_epochs - 1:
        continue

    # Save the model:
    with open(output_folder + f"/checkpoints/e{epoch_num:06}.model", "wb") as file:
        pickle.dump(state, file)

    pred_folders = {
        "train": output_folder + f"/predictions/train/e{epoch_num:06}",
        "val": output_folder + f"/predictions/val/e{epoch_num:06}",
    }

    # Create folders
    os.makedirs(pred_folders["train"])
    os.makedirs(pred_folders["val"])

    # Evaluate:

    # (1) image grid:
    fig, axs = plt.subplots(12, 4, figsize=(8.4, 2 * 12.8))

    mean_mse_train = 0
    mean_psnr_train = 0
    mean_ssim_train = 0
    mean_lpips_train = 0
    train_imgs = train_data.map(
        lambda datum: jnp.clip(render_image(state, datum, jax.random.key(0)), 0.0, 1.0),
        batch_size=args.batch_size_render,
    )
    for i in range(len(train_data)):
        train_datum = train_data[i]
        train_img = train_imgs[i]
        if i < 12:
            axs[i, 0].imshow(train_img)
            axs[i, 1].imshow(train_datum.pixels)

        train_img = np.array(train_img)
        out_img = im.fromarray(np.array(255.0 * np.array(train_img), dtype=np.uint8))
        out_img.save(f"{pred_folders['train']}/{i:06}.png")

        mean_mse_train += jnp.mean((train_img - train_datum.pixels) ** 2) / len(
            train_data
        )
        mean_psnr_train += peak_signal_noise_ratio(train_img, train_datum.pixels) / len(
            train_data
        )
        mean_ssim_train += structural_similarity(
            train_img, train_datum.pixels, channel_axis=-1, data_range=1.0
        ) / len(train_data)
        mean_lpips_train += lpips(train_img, train_datum.pixels) / len(train_data)

    mean_mse_val = 0
    mean_psnr_val = 0
    mean_ssim_val = 0
    mean_lpips_val = 0
    val_imgs = val_data.map(
        lambda datum: jnp.clip(render_image(state, datum, jax.random.key(0)), 0.0, 1.0),
        batch_size=args.batch_size_render,
    )
    for i in range(len(val_data)):
        val_datum = val_data[i]
        val_img = val_imgs[i]
        if i < 12:
            axs[i, 2].imshow(val_img)
            axs[i, 3].imshow(val_datum.pixels)

        out_img = im.fromarray(np.array(255.0 * np.array(val_img), dtype=np.uint8))
        out_img.save(f"{pred_folders['val']}/{i:06}.png")

        mean_mse_val += jnp.mean((val_img - val_datum.pixels) ** 2) / len(val_data)
        mean_psnr_val += peak_signal_noise_ratio(val_img, val_datum.pixels) / len(
            val_data
        )
        mean_ssim_val += structural_similarity(
            val_img, val_datum.pixels, channel_axis=-1, data_range=1.0
        ) / len(val_data)
        mean_lpips_val += lpips(val_img, val_datum.pixels) / len(val_data)

    for ax in np.ravel(axs):
        ax.axis("off")
    axs[0, 0].set_title("train rendered")
    axs[0, 1].set_title("train ground truth")
    axs[0, 2].set_title("val rendered")
    axs[0, 3].set_title("val ground truth")
    fig.savefig(f"{output_folder}/image_grid/{epoch_num:04}.png")
    plt.close()

    # (2) density field dump:
    ii, jj, kk = jnp.meshgrid(
        jnp.linspace(-args.grid_radius, args.grid_radius, 200),
        jnp.linspace(-args.grid_radius, args.grid_radius, 200),
        jnp.linspace(-args.grid_radius, args.grid_radius, 200),
    )
    points = jnp.stack((np.ravel(ii), np.ravel(jj), np.ravel(kk)), axis=1)
    dirs = jax.random.uniform(
        jax.random.key(0), shape=(points.shape[0], 3), minval=-1.0, maxval=+1.0
    )
    densities = jax.vmap(lambda x: model.density_function(state, x))(points).reshape(
        ii.shape
    )
    colors = jax.vmap(lambda x, dir: model.color_function(state, x, dir))(points, dirs)
    ic(jnp.min(densities))
    ic(jnp.max(densities))
    ic(jnp.min(colors, axis=0))
    ic(jnp.max(colors, axis=0))
    colors = colors.reshape(ii.shape + (3,))
    np.save(f"{output_folder}/predictions/density_dump-{epoch_num:04}.npy", densities)

    # (3) PSNR learning curves:
    train_mses.append(float(mean_mse_train))
    train_psnrs.append(float(mean_psnr_train))
    train_ssims.append(float(mean_ssim_train))
    train_lpips.append(float(mean_lpips_train))
    val_mses.append(float(mean_mse_val))
    val_psnrs.append(float(mean_psnr_val))
    val_ssims.append(float(mean_ssim_val))
    val_lpips.append(float(mean_lpips_val))

    density_norms.append(float(model.representation.density_norm(gradient_approx)))
    color_norms.append(float(model.representation.color_norm(gradient_approx)))

    plt.figure()
    plt.title(f"3D, n_train={len(train_data)}, img_shape={train_data[0].pixels.shape}")
    plt.plot(train_mses, "-o", label="Train mse")
    plt.plot(val_mses, "-o", label="val mse")
    plt.xlabel("epochs")
    plt.ylabel("mse")
    plt.legend(loc="best")
    plt.savefig(f"{output_folder}/mse.png")
    plt.close()

    plt.figure()
    plt.title(f"3D, n_train={len(train_data)}, img_shape={train_data[0].pixels.shape}")
    plt.plot(train_psnrs, "-o", label="Train PSNR")
    plt.plot(val_psnrs, "-o", label="val PSNR")
    plt.xlabel("epochs")
    plt.ylabel("PSNR")
    plt.legend(loc="best")
    plt.savefig(f"{output_folder}/psnr.png")
    plt.close()

    plt.figure()
    plt.title(f"3D, n_train={len(train_data)}, img_shape={train_data[0].pixels.shape}")
    plt.plot(train_ssims, "-o", label="Train SSIM")
    plt.plot(val_ssims, "-o", label="val SSIM")
    plt.xlabel("epochs")
    plt.ylabel("SSIM")
    plt.legend(loc="best")
    plt.savefig(f"{output_folder}/ssim.png")
    plt.close()

    plt.figure()
    plt.title(f"3D, n_train={len(train_data)}, img_shape={train_data[0].pixels.shape}")
    plt.plot(train_lpips, "-o", label="Train LPIPS")
    plt.plot(val_lpips, "-o", label="val LPIPS")
    plt.xlabel("epochs")
    plt.ylabel("LPIPS")
    plt.legend(loc="best")
    plt.savefig(f"{output_folder}/lpips.png")
    plt.close()

    with atomic_write(f"{output_folder}/mse.json") as file:
        json.dump(
            {
                "train": train_mses,
                "val": val_mses,
            },
            file,
        )

    with atomic_write(f"{output_folder}/psnr.json") as file:
        json.dump(
            {
                "train": train_psnrs,
                "val": val_psnrs,
            },
            file,
        )

    with atomic_write(f"{output_folder}/ssim.json") as file:
        json.dump(
            {
                "train": train_ssims,
                "val": val_ssims,
            },
            file,
        )

    with atomic_write(f"{output_folder}/lpips.json") as file:
        json.dump(
            {
                "train": train_lpips,
                "val": val_lpips,
            },
            file,
        )

    plt.figure()
    plt.title(
        f"Grad norms, n_train={len(train_data)}, img_shape={train_data[0].pixels.shape}"
    )
    plt.plot(density_norms, "-o", label="Grad norm wrt. density")
    plt.plot(color_norms, "-o", label="Grad norm wrt. density")
    plt.xlabel("epochs")
    plt.ylabel("Grad norms")
    plt.legend(loc="best")
    plt.savefig(f"{output_folder}/grad_norms.png")
    plt.close()

    with atomic_write(f"{output_folder}/grad_norms.json") as file:
        json.dump(
            {
                "grad_norm_wrt_density": density_norms,
                "grad_norm_wrt_color": color_norms,
            },
            file,
        )

    if jnp.isnan(mean_psnr_train) or jnp.isnan(mean_psnr_val):
        print("PSNR is NaN! Exiting.")
        break

end_time = time.time()

with open(f"{output_folder}/runtime.txt", "w") as file:
    print(f"{end_time - start_time} seconds", file=file)
    print(f"{(end_time - start_time) / 60} minutes", file=file)
    print(f"{(end_time - start_time) / 60 / 60} hours", file=file)
