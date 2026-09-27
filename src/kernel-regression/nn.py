import json
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from data import get_codrna_data
from sklearn.model_selection import train_test_split

DEFAULT_MAX_SAMPLES = 500_000
DEFAULT_N_EPOCHS = 60
DEFAULT_HIDDEN_DIMS = [256, 256]

parser = ArgumentParser(description="Run NN baseline on UCI sepsis.")
parser.add_argument(
    "--max-samples",
    type=int,
    default=DEFAULT_MAX_SAMPLES,
    help="Maximum number of samples to subsample from the full dataset.",
)
parser.add_argument(
    "--n-epochs",
    type=int,
    default=DEFAULT_N_EPOCHS,
    help="Number of optimization steps.",
)
parser.add_argument(
    "--hidden-dims",
    type=str,
    default=",".join(str(x) for x in DEFAULT_HIDDEN_DIMS),
    help="Comma-separated hidden layer dimensions (e.g. 256,256).",
)
parser.add_argument(
    "--log-every",
    type=int,
    default=10,
    help="Logging frequency in epochs.",
)
parser.add_argument(
    "--results-file",
    type=Path,
    required=True,
    help="Output JSON path.",
)
parser.add_argument(
    "--use-engineered-features",
    dest="use_engineered_features",
    action="store_true",
    default=False,
    help="Use engineered time/cyclical features from prepared dataset.",
)
parser.add_argument(
    "--no-engineered-features",
    dest="use_engineered_features",
    action="store_false",
    help="Use only base (non-engineered) features (default).",
)
parser.add_argument("--loss", choices=["mse", "bce"], required=True)
args = parser.parse_args()

SEED = 0
TEST_SIZE = 0.2

LOSS = args.loss
match LOSS:
    case "mse":
        LR = 1e-3
    case "bce":
        LR = 1e-4


def parse_hidden_dims(raw: str) -> list[int]:
    dims = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not dims:
        raise ValueError("hidden_dims cannot be empty")
    if any(d <= 0 for d in dims):
        raise ValueError("all hidden dimensions must be > 0")
    return dims


class MLP(nn.Module):
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        for layer_size in self.layers:
            x = nn.Dense(layer_size)(x)
            x = jax.nn.gelu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


def main(
    *,
    use_engineered_features: bool,
    max_samples: int,
    n_epochs: int,
    lr: float,
    hidden_dims: list[int],
    results_file: Path,
    log_every: int,
) -> None:
    np.random.seed(SEED)

    x, y = get_codrna_data()
    x = x[:, :3]

    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(x),
        np.asarray(y),
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    x_train = jnp.array(x_train, dtype=jnp.float32)
    x_test = jnp.array(x_test, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)

    mlp = MLP(layers=hidden_dims)

    def objective_loss(params, x: jax.Array, y: jax.Array) -> jax.Array:
        preds = mlp.apply(params, x)
        match LOSS:
            case "mse":
                return 0.5 * jnp.mean(jnp.square(preds - y))
            case "bce":
                return -jnp.mean(
                    y * jax.nn.log_sigmoid(preds) + (1 - y) * jax.nn.log_sigmoid(-preds)
                )
            case _:
                raise RuntimeError()

    @jax.jit
    def predict(params, x: jax.Array) -> jax.Array:
        return mlp.apply(params, x)

    params = mlp.init(jax.random.key(SEED), jnp.empty((x.shape[1],)))
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    objective_loss_and_grad = jax.jit(jax.value_and_grad(objective_loss))

    # Warmup JIT paths so compile cost is outside train_time_seconds.
    warm_loss, warm_Grads = objective_loss_and_grad(params, x_train, y_train)
    warm_test_loss = objective_loss(params, x_test, y_test)
    warm_train_preds = predict(params, x_train)
    warm_test_preds = predict(params, x_test)
    jax.block_until_ready(warm_loss)
    jax.block_until_ready(warm_test_loss)
    jax.block_until_ready(warm_train_preds)
    jax.block_until_ready(warm_test_preds)

    train_loss_history: list[float] = [
        float(objective_loss_and_grad(params, x_train, y_train)[0])
    ]
    test_loss_history: list[float] = [float(objective_loss(params, x_test, y_test))]
    train_mse_history: list[float] = []
    test_mse_history: list[float] = []
    elapsed_time_seconds: list[float] = []
    step_history: list[int] = []

    before = time.time()
    for epoch_num in range(n_epochs):
        train_loss, loss_grads = objective_loss_and_grad(params, x_train, y_train)

        updates, optimizer_state = optimizer.update(loss_grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        test_loss = objective_loss(params, x_test, y_test)

        train_loss_value = float(train_loss)
        test_loss_value = float(test_loss)
        elapsed = time.time() - before

        train_loss_history.append(train_loss_value)
        test_loss_history.append(test_loss_value)
        train_mse_history.append(2.0 * train_loss_value)
        test_mse_history.append(2.0 * test_loss_value)
        elapsed_time_seconds.append(elapsed)
        step_history.append(epoch_num + 1)

        if epoch_num % log_every == 0 or epoch_num == n_epochs - 1:
            print(
                f"epoch={epoch_num:03d} train_loss={train_loss_value:.6f} "
                f"test_loss={test_loss_value:.6f}"
            )

    # Ensure training queue finished before timing cutoff.
    train_final_probe = objective_loss(params, x_train, y_train)
    jax.block_until_ready(train_final_probe)
    train_time_sec = time.time() - before

    pred_before = time.time()
    train_preds = predict(params, x_train)
    test_preds = predict(params, x_test)
    jax.block_until_ready(test_preds)
    pred_time_sec = time.time() - pred_before

    train_mse = float(objective_loss(params, x_train, y_train))
    test_mse = float(objective_loss(params, x_test, y_test))

    results = {
        "method": "Neural Network",
        "dataset": "uci_sepsis",
        "metrics": {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_time_seconds": train_time_sec,
            "pred_time_seconds": pred_time_sec,
        },
        "config": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "max_samples": max_samples,
            "n_epochs": n_epochs,
            "lr": lr,
            "hidden_dims": hidden_dims,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "n_features": int(x_train.shape[1]),
            "use_engineered_features": bool(use_engineered_features),
        },
        "learning_curve": {
            "step": step_history,
            "elapsed_time_seconds": elapsed_time_seconds,
            "train_loss": train_loss_history,
            "test_loss": test_loss_history,
            "train_mse": train_mse_history,
            "test_mse": test_mse_history,
            "loss_definition": "0.5 * mean((pred - y)^2)",
        },
    }

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"Total train time: {train_time_sec:.3f}s")
    print(f"Final train MSE: {train_mse:.6f}")
    print(f"Final test  MSE: {test_mse:.6f}")
    print(f"Prediction time (train+test): {pred_time_sec:.3f}s")
    print(f"Saved metrics JSON to: {results_file}")


main(
    use_engineered_features=args.use_engineered_features,
    max_samples=args.max_samples,
    n_epochs=args.n_epochs,
    lr=LR,
    hidden_dims=parse_hidden_dims(args.hidden_dims),
    results_file=args.results_file,
    log_every=max(1, args.log_every),
)
