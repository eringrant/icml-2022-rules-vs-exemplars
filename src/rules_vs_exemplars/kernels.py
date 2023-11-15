import gin
import jax.numpy as jnp


@gin.configurable
def gaussian(x1, x2):
  return jnp.exp(-jnp.sum((x1 - x2) ** 2))


@gin.configurable
def linear(x1, x2):
  return jnp.dot(x1.T, x2)


# learning takes params to nan
# def tanh(x1, x2):
# return jnp.tanh(jnp.dot(x1.T, x2))


@gin.configurable
def cauchy(x1, x2):
  return 1.0 / (1.0 + jnp.sum((x1 - x2) ** 2))

@gin.configurable
def ornsteinuhlenbeck(x1, x2):
  return jnp.exp(-jnp.sum(jnp.abs(x1 - x2)))

@gin.configurable
def quadratic(x1, x2):
  return (1.0 + jnp.dot(x1.T, x2))**2

@gin.configurable
def cubic(x1, x2):
  return (1.0 + jnp.dot(x1.T, x2))**3
