import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Callable, Protocol, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@contextmanager
def atomic_write(target_path: PathLike):
    tmp_folder = Path("tmp")
    tmp_folder.mkdir(exist_ok=True)
    tmp_path = tmp_folder / str(uuid.uuid4())
    with tmp_path.open("w") as tmpfile:
        yield tmpfile
        os.rename(tmpfile.name, target_path)


@contextmanager
def timed(description: str):
    print(f"{description}... ", end="", flush=True)
    before = time.time()
    yield
    after = time.time()
    delta = after - before
    if delta >= 60:
        print(f"done in {(after - before) / 60:.2f}mins")
    else:
        print(f"done in {after - before:.2f}secs")


def viewing_direction_to_unit_circle(
    direction: Float[Array, "d"],
) -> Float[Array, "d-1"]:
    """
    Convert a vector in R^d to coordinates of its projection's representation on the unit hypersphere in R^d, S^(d-1).
    """

    # project to the unit hypersphere (i.e., normalize):
    direction = direction / jnp.linalg.norm(direction)

    # figure out their coordinates:
    if direction.shape == (2,):
        u = jnp.acos(direction[0])
        return jnp.array([u])
    elif direction.shape == (3,):
        u = jnp.acos(direction[2])
        v = jnp.acos(
            jnp.clip(direction[0] / jnp.sin(u), -1.0, 1.0)
        )  # clip to avoid numerical precision issues
        return jnp.array([u, v])
    else:
        raise ValueError(
            f"unhandled shape for `viewing_direction_to_unit_circle`: {direction.shape}"
        )


class BaseLearner(Protocol):
    """
    Type descriptor for Scikit-Learn-type base learners.
    """

    def fit(
        self, x: Float[Array, "n p"], y: Float[Array, "n c"] | Float[Array, "n"]
    ) -> Self | None:
        pass

    def predict(
        self, x: Float[Array, "n p"]
    ) -> Float[Array, "n c"] | Float[Array, "n"]:
        pass


def jax_predict(model: BaseLearner, *, output_dim: int):
    """
    Wrapper for calling Scikit-Learn models (which do not use JAX) from within JAX, in particular allowing jax.vmap and jax.jit to work.

    Use like: `jax_predict(model, output_dim=1)(input_data)`.
    """

    def inner(x):
        def predict(x_inner):
            import numpy as np

            x_inner = np.asarray(x_inner, copy=False)

            *original_shape, p = x_inner.shape
            reshaped = x_inner.reshape(-1, p)
            preds = model.predict(reshaped)
            if preds.ndim == 1:
                return preds.reshape(original_shape).astype(np.float32)
            else:
                return preds.reshape(original_shape + [preds.shape[1]]).astype(
                    np.float32
                )

        match x.shape:
            case (p,):
                if output_dim == 1:
                    out_shape = jax.ShapeDtypeStruct((), jnp.float32)
                else:
                    out_shape = jax.ShapeDtypeStruct((output_dim,), jnp.float32)
            case (n, p):
                if output_dim == 1:
                    out_shape = jax.ShapeDtypeStruct(
                        (n,),
                        jnp.float32,
                    )
                else:
                    out_shape = jax.ShapeDtypeStruct(
                        (
                            n,
                            output_dim,
                        ),
                        jnp.float32,
                    )
            case _:
                raise ValueError(f"{x.shape = }")
        return jax.pure_callback(predict, out_shape, x, vmap_method="broadcast_all")

    return inner


def integral(
    start: float | Float[Array, ""],
    stop: float | Float[Array, ""],
    func: Callable[[Float[Array, ""]], Float[Array, "b"]],
    *,
    n_samples: int = 100,
) -> Float[Array, "b"]:
    """
    Approximate the integral of func in `[start, stop]`, using `n_samples` samples. Currently does a Riemann sum (i.e., not very smart).
    """

    ts = jnp.linspace(start, stop, n_samples, endpoint=False)
    values = jax.vmap(func)(ts)
    return jnp.mean(values, axis=0) * (stop - start)


def cumulative_trapezoid(y, *, dx, axis=None):
    # JAX version of https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html#scipy.integrate.cumulative_trapezoid
    # Slightly different API and computation though
    return jnp.insert(
        0.5 * dx * jnp.cumsum(y[1:] + y[:-1], axis=axis),
        0,  # at pos 0
        0.0,  # value 0.0
        axis=axis,
    )


type JaxRng = Array


def repr_lambda(obj):
    import inspect
    import textwrap

    return textwrap.dedent(inspect.getsource(obj)).strip()
