import jax.numpy as jnp
from jaxtyping import Array, Float


def f_star(x: Float[Array, "2"]) -> Float[Array, ""]:
    x, y = x

    omega = 20.0
    m = 3
    sigma = 0.5

    xx = 1.5 * x - 0.75
    yy = 1.5 * y - 0.75

    r = jnp.sqrt(xx**2 + yy**2)
    theta = jnp.arctan2(yy, xx)

    # The phase couples distance and angle to create the spiral
    phase = omega * r - m * theta
    envelope = jnp.exp(-(r**2) / (sigma**2))

    return jnp.cos(phase) * envelope
