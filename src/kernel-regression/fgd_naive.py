import json
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from icecream import ic
from jaxtyping import Array, Float
from data import (
    get_codrna_data,
    get_gas_turbine_data,
    get_nutrition_data,
    get_sepsis_data,
    get_synthetic_data,
)
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument("--depth", type=int, required=True)
parser.add_argument("--results-file", required=True)
parser.add_argument("--loss", choices=["mse", "bce"], required=True)
args = parser.parse_args()

TEST_SIZE = 0.2
MAX_SAMPLES = 500_000

N_EPOCHS = 60
match args.loss:
    case "mse":
        LR = 10
    case "bce":
        LR = 50
EPS = 0.5
DEPTH = args.depth
RBF_GAMMA = 100.0

BATCH_SIZE_LEAF = 2**4
BATCH_SIZE_DATA = 2**20

LOSS: Literal["mse", "bce"] = args.loss

RESULTS_FILE = Path(args.results_file)


@jdc.pytree_dataclass
class Representation:
    split_features: Float[Array, "num_internal"]
    split_thresholds: Float[Array, "num_internal"]
    leaf_values: Float[Array, "num_leaves"]
    leaf_mins: Float[Array, "num_leaves d"]
    leaf_maxs: Float[Array, "num_leaves d"]

    @jax.jit
    def __call__(self, x):
        # Bypass JAX tracing the loop body when the tree has 0 depth.
        if self.split_features.shape[0] == 0:
            return self.leaf_values[0]

        def body_fn(i, node_idx):
            feat = self.split_features[node_idx].astype(jnp.int32)
            thresh = self.split_thresholds[node_idx]
            go_left = x[feat] <= thresh
            # Left child is 2*i + 1, Right child is 2*i + 2
            return 2 * node_idx + 1 + jnp.where(go_left, 0, 1)

        # The depth of the tree is log2 of the number of leaves
        depth = jnp.log2(self.leaf_values.shape[0]).astype(jnp.int32)

        # Traverse from root to leaf
        final_node_idx = jax.lax.fori_loop(0, depth, body_fn, 0)

        # Map complete tree index back to the leaf array
        leaf_idx = final_node_idx - (self.leaf_values.shape[0] - 1)
        return self.leaf_values[leaf_idx]

    @classmethod
    def zeros(cls, d: int = 2):
        # Now accepts `d` to initialize bounds for arbitrary dimensions
        return cls(
            split_features=jnp.zeros((0,), dtype=jnp.int32),
            split_thresholds=jnp.zeros((0,), dtype=jnp.float32),
            leaf_values=jnp.zeros((1,), dtype=jnp.float32),
            leaf_mins=jnp.zeros((1, d), dtype=jnp.float32),
            leaf_maxs=jnp.ones((1, d), dtype=jnp.float32),
        )

    @jax.jit
    def subdivide(self):
        n_leaves, d = self.leaf_mins.shape

        # Find the widest dimension for each leaf to split on
        widths = self.leaf_maxs - self.leaf_mins
        split_feats = jnp.argmax(widths, axis=1)

        # Threshold is the midpoint of the chosen dimension
        idx = jnp.arange(widths.shape[0])
        split_threshs = (
            self.leaf_mins[idx, split_feats] + widths[idx, split_feats] / 2.0
        )

        # Old leaves become internal nodes; append them to the tree arrays
        new_split_features = jnp.concatenate([self.split_features, split_feats])
        new_split_thresholds = jnp.concatenate([self.split_thresholds, split_threshs])

        # Duplicate the values for the two new children of each split leaf
        new_leaf_values = jnp.repeat(self.leaf_values, 2)

        # Calculate bounding boxes for the new children
        left_maxs = self.leaf_maxs.at[idx, split_feats].set(split_threshs)
        right_mins = self.leaf_mins.at[idx, split_feats].set(split_threshs)

        # Interleave the child bounds to match the binary tree indexing scheme
        new_leaf_mins = jnp.empty((2 * n_leaves, d))
        new_leaf_mins = new_leaf_mins.at[0::2].set(self.leaf_mins)
        new_leaf_mins = new_leaf_mins.at[1::2].set(right_mins)

        new_leaf_maxs = jnp.empty((2 * n_leaves, d))
        new_leaf_maxs = new_leaf_maxs.at[0::2].set(left_maxs)
        new_leaf_maxs = new_leaf_maxs.at[1::2].set(self.leaf_maxs)

        return jdc.replace(
            self,
            split_features=new_split_features,
            split_thresholds=new_split_thresholds,
            leaf_values=new_leaf_values,
            leaf_mins=new_leaf_mins,
            leaf_maxs=new_leaf_maxs,
        )

    @jax.jit
    def __add__(self, other: "Representation"):
        return jdc.replace(self, leaf_values=self.leaf_values + other.leaf_values)

    @jax.jit
    def __sub__(self, other: "Representation"):
        return jdc.replace(self, leaf_values=self.leaf_values - other.leaf_values)

    @jax.jit
    def __mul__(self, other: float):
        return jdc.replace(self, leaf_values=other * self.leaf_values)

    @jax.jit
    def __rmul__(self, other: float):
        return jdc.replace(self, leaf_values=self.leaf_values * other)

    @partial(jax.jit, static_argnames=("f",))
    def approx(
        self,
        f: Callable[
            ["Representation", Float[Array, "n"], Float[Array, "d"]], Float[Array, ""]
        ],
        *,
        l2_residuals: Float[Array, "n"],
        loss_residuals: Float[Array, "n"],
    ) -> tuple["Representation", Float[Array, ""]]:
        n_leaves, d = self.leaf_mins.shape
        offsets = 0.5 * jnp.ones((8, d))

        def fit_leaf(args):
            start, end = args
            points = jax.vmap(lambda offset: start + offset * (end - start))(offsets)
            samples = jax.vmap(lambda x: f(self, loss_residuals, x))(points)
            return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)

        # Map the sampling function over the bounding boxes of all leaves
        new_values, approx_errors = jax.lax.map(
            fit_leaf,
            (self.leaf_mins, self.leaf_maxs),
            batch_size=BATCH_SIZE_LEAF,
        )

        norm2 = jnp.sum(new_values**2)
        rel_error2 = jnp.sum(approx_errors) / norm2

        new_rep = jdc.replace(self, leaf_values=new_values)
        return new_rep, jnp.sqrt(rel_error2)


x_full, y_full = get_codrna_data()
x_full = x_full[:, :3]
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=TEST_SIZE, random_state=0
)

mins, maxs = jnp.min(x_train, axis=0), jnp.max(x_train, axis=0)
ic(mins, maxs)
x_train = (x_train - mins) / (maxs - mins)
x_test = (x_test - mins) / (maxs - mins)


def kernel(a: Float[Array, "d"], b: Float[Array, "d"]) -> Float[Array, ""]:
    diff = a - b
    return jnp.exp(-RBF_GAMMA * jnp.dot(diff, diff))


@jax.jit
def loss(
    f: Representation, x: Float[Array, "n p"], y: Float[Array, "n"]
) -> Float[Array, ""]:
    preds = jax.lax.map(f, x, batch_size=BATCH_SIZE_DATA)
    match LOSS:
        case "mse":
            return 0.5 * jnp.mean((preds - y) ** 2)
        case "bce":
            return -jnp.mean(
                y * jax.nn.log_sigmoid(preds) + (1 - y) * jax.nn.log_sigmoid(-preds)
            )
        case _:
            raise RuntimeError()


@jax.jit
def loss_pregrad(f: Representation) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    preds = jax.lax.map(f, x_train, batch_size=BATCH_SIZE_DATA)
    l2_residuals = preds - y_train
    match LOSS:
        case "mse":
            loss_residuals = preds - y_train
        case "bce":
            loss_residuals = -jax.vmap(
                jax.grad(
                    lambda pred, this_y: this_y * jax.nn.log_sigmoid(pred)
                    + (1 - this_y) * jax.nn.log_sigmoid(-pred)
                )
            )(preds, y_train)
    return l2_residuals, loss_residuals


def loss_grad(
    f: Representation, loss_residuals: Float[Array, "n"], x: Float[Array, "d"]
) -> Float[Array, ""]:
    weights = jax.lax.map(lambda x_: kernel(x_, x), x_train, batch_size=BATCH_SIZE_DATA)
    return jnp.mean(loss_residuals * weights)


n_features = int(x_train.shape[1])
f = Representation.zeros(x_train.shape[1])

train_loss_history: list[float] = [float(loss(f, x_train, y_train))]
test_loss_history: list[float] = [float(loss(f, x_test, y_test))]
elapsed_time_seconds: list[float] = []

for _ in range(DEPTH):
    f = f.subdivide()

print("JIT warmup")
l2_residuals, loss_residuals = loss_pregrad(f)
g, rel_error = f.approx(
    loss_grad, loss_residuals=loss_residuals, l2_residuals=l2_residuals
)
f - LR * g
print("JIT warmup done")

before = time.time()
for epoch_num in range(N_EPOCHS):
    l2_residuals, loss_residuals = loss_pregrad(f)

    g, rel_error = f.approx(
        loss_grad, loss_residuals=loss_residuals, l2_residuals=l2_residuals
    )
    print(f"initial rel_error={float(rel_error):.4f}")

    f = f - LR * g

    train_loss_value = float(loss(f, x_train, y_train))
    test_loss_value = float(loss(f, x_test, y_test))
    elapsed = time.time() - before

    train_loss_history.append(train_loss_value)
    test_loss_history.append(test_loss_value)
    elapsed_time_seconds.append(elapsed)

    if epoch_num % 10 == 0 or epoch_num == N_EPOCHS - 1:
        print(
            f"epoch={epoch_num:03d} rel_err={float(rel_error):.4f} "
            f"train_loss={train_loss_value:.6f} test_loss={test_loss_value:.6f}"
        )

jax.block_until_ready(f)
after = time.time()
train_time_sec = after - before

results = {
    "method": "Ours",
    "learning_curve": {
        "elapsed_time_seconds": elapsed_time_seconds,
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
    },
}

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=2)
