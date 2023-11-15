"""Models for partial exposure simulations."""
from functools import partial
from typing import Callable, Iterable, Optional, Mapping, Any
from typing import NamedTuple

import gin
import tensorflow as tf

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, cho_solve, solve
from jax.scipy.special import erf, expit
import haiku as hk

from rules_vs_exemplars import kernels
from rules_vs_exemplars.data import Batch

from vit_jax import models as vit_jax_models

# Amortized computations for GP classifier approximation:
# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = jnp.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, jnp.newaxis]
COEFS = jnp.array(
  [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, jnp.newaxis]


class Prediction(NamedTuple):
  logits: jnp.ndarray


class GPPrediction(NamedTuple):
  logits: jnp.ndarray
  lml: jnp.ndarray


@jax.jit
def one_hot_accuracy(predictions: jnp.ndarray, batch: Batch) -> jnp.ndarray:
  # labels = jax.nn.one_hot(batch['labels'], 1000)
  avg_acc = jnp.mean(jnp.argmax(predictions, axis=-1) == batch.target)
  return avg_acc


@jax.jit
def xent_loss(
  predictions: Prediction,
  onehot_labels: jnp.ndarray,
) -> jnp.ndarray:
  return -jnp.sum(
    onehot_labels * jax.nn.log_softmax(predictions.logits), axis=-1
  )


@jax.jit
def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
  return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def nll_loss(
  predictions: Prediction,
  onehot_labels: jnp.ndarray,
) -> jnp.ndarray:
  """
  negative log marginal likelihood of model hyperparameters

  Unaffected by test data, this just checks how well
  the current hyperparams fit training data.

  Applies only to GP since the hyperparameter optimization is the
  only GD-optimized part. The fit to training data is approximated
  with newton iteration (see `rules_vs_exemplars.models`).

  """
  return -predictions.lml


class PredictionWrapper(object):
  def __call__(self, x, is_training):
    try:
      return Prediction(super().__call__(x, is_training=is_training))
    except TypeError as e:
      if "is_training" in str(e):
        return Prediction(super().__call__(x))
      else:
        raise e


@gin.configurable
class MLP(PredictionWrapper, hk.nets.MLP):
  """A multi-layer perceptron module with fixed default size."""

  loss = staticmethod(xent_loss)
  accuracy = staticmethod(one_hot_accuracy)

  def __init__(
    self,
    layers: Iterable[int],
    n_layers: int,
    n_hiddens: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
  ):
    if not layers:
      output_sizes = n_layers * [n_hiddens] + [2]
    else:
      output_sizes = layers + [2]
    super().__init__(output_sizes)
    self.activation = activation


@gin.configurable
class LogReg(PredictionWrapper, hk.nets.MLP):
  """Logistic regression derived from an MLP with no hiddens."""

  loss = staticmethod(xent_loss)
  accuracy = staticmethod(one_hot_accuracy)

  def __init__(self, output_sizes: Iterable[int] = [2]):
    super().__init__(output_sizes)


@gin.configurable
class GP(hk.Module):
  """Gaussian process classifier.

  Adapted from the scikit-learn implementation:
  https://github.com/scikit-learn/scikit-learn/blob/7cc3dbcbe/sklearn/gaussian_process/_gpc.py#L459

  as well as JAX-friendly implementation from:
  https://github.com/ExpectationMax/sklearn-jax-kernels/blob/master/sklearn_jax_kernels/gpc.py

  The optimizable hyperparameters are length scale, variance, and the
  observation noise.
  """

  loss = staticmethod(nll_loss)
  accuracy = staticmethod(one_hot_accuracy)

  def __init__(
    self,
    ls: float,
    max_iter_predict: int,
    kernel: Callable,
    base_data: tf.data.Dataset,
  ):
    super().__init__()
    self.kernel = kernel
    self.max_iter_predict = max_iter_predict
    self.ls = ls
    self.fit_ls = kernel != kernels.linear

    # Batch and repeat the entire base dataset.
    base_data = base_data.batch(len(base_data))
    base_data = base_data.repeat()
    self.base_data = next(base_data.as_numpy_iterator())

  def __call__(self, X: jnp.ndarray, is_training: bool) -> GPPrediction:
    """Forward pass."""
    del is_training

    # amplitude and noise fitting reflects model misspecification
    # high amplitude / low noise always fits better, ill defined
    # fixing these to constants

    # noise_init = hk.initializers.Constant(0.0)
    # noise = self.softplus(hk.get_parameter("noise",
    # shape=[1, 1],
    # init=noise_init))

    # amp_init = hk.initializers.Constant(10*self.ls)
    # amp = self.softplus(hk.get_parameter("amp",
    # shape=[1, 1],
    # init=amp_init))

    amp = noise = 1.0

    # ls is well-defined only for gaussian not linear
    if self.fit_ls:
      ls_init = hk.initializers.Constant(self.ls)
      ls = self.softplus(hk.get_parameter("ls", shape=[1, 1], init=ls_init))
    else:
      ls = 1.0

    xtest = X / ls
    # downstream expects y to be labels, not one-hot.
    # added hack that is hardcoded for 2 classes.
    x, y = self.base_data.examples / ls, self.base_data.targets[:, 1]
    eye = jnp.eye(y.size)

    K = amp * self.cov_map(self.kernel, x) + eye * (noise + 1e-6)

    # need to recompute each time in case hyperparams have changed
    # could consider memoizing

    # Use Newton's iteration method to find mode of Laplace approximation
    # Algoritm 3.1 in R/W GPML
    f = jnp.zeros_like(y, dtype=jnp.float32)
    newton_iteration = partial(self._newton_iteration, y, K)
    lml, f, (pi, W_sr, L, b, a) = newton_iteration(f)

    # log_marginal_likelihood = -jnp.inf
    for _ in range(self.max_iter_predict):
      lml, f, (pi, W_sr, L, b, a) = newton_iteration(f)
      # Check if we have converged (log marginal likelihood does
      # not decrease)
      # JAX doesn't permit dependency on state, remove convergence criteria
      # always runs for max_iter
      # extra slow...
      # if lml - log_marginal_likelihood < 1e-10:
      # break

      # log_marginal_likelihood = lml

    # As discussed on Section 3.4.2 of GPML, for making hard binary
    # decisions, it is enough to compute the MAP of the posterior and
    # pass it through the link function
    K_star = amp * self.cov_map(self.kernel, x, xtest)
    f_star = K_star.T.dot(y - pi)  # Algorithm 3.2,Line 4

    v = solve(L, W_sr[:, jnp.newaxis] * K_star)  # Line 5
    # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
    K_test = amp * self.cov_map(self.kernel, xtest)
    var_f_star = jnp.diag(K_test) - jnp.einsum("ij,ij->j", v, v)

    # Line 7:
    # Approximate \int log(z) * N(z | f_star, var_f_star)
    # Approximation is due to Williams & Barber, "Bayesian Classification
    # with Gaussian Processes", Appendix A: Approximate the logistic
    # sigmoid by a linear combination of 5 error functions.
    # For information on how this integral can be computed see
    # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
    alpha = 1 / (2 * var_f_star)
    gamma = LAMBDAS * f_star
    integrals = (
      jnp.sqrt(jnp.pi / alpha)
      * erf(gamma * jnp.sqrt(alpha / (alpha + LAMBDAS ** 2)))
      / (2 * jnp.sqrt(var_f_star * 2 * jnp.pi))
    )
    pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

    return GPPrediction(jnp.vstack((1 - pi_star, pi_star)).T, lml)

  def softplus(self, x):
    return jnp.logaddexp(x, 0.0)

  def cov_map(self, cov_func, xs, xs2=None):
    """Compute a covariance matrix from a covariance function and data points.

    Args:
      cov_func: callable function, maps pairs of data points to scalars.
      xs: array of data points, stacked along the leading dimension.

    Returns:
      A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
    """
    if xs2 is None:
      return jax.vmap(lambda x: jax.vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
      return jax.vmap(lambda x: jax.vmap(lambda y: cov_func(x, y))(xs))(xs2).T

  @partial(jax.jit, static_argnums=0)
  def _newton_iteration(self, y_train, K, f):
    pi = expit(f)
    W = pi * (1 - pi)
    # Line 5
    W_sr = jnp.sqrt(W)
    W_sr_K = W_sr[:, jnp.newaxis] * K
    B = jnp.eye(W.shape[0]) + W_sr_K * W_sr
    L = cholesky(B, lower=True)
    # Line 6
    b = W * f + (y_train - pi)
    # Line 7
    a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
    # Line 8
    f = K.dot(a)

    # Line 10: Compute log marginal likelihood in loop and use as
    #          convergence criterion
    lml = (
      -0.5 * a.T.dot(f)
      - jnp.log1p(jnp.exp(-(y_train * 2 - 1) * f)).sum()
      - jnp.log(jnp.diag(L)).sum()
    )
    return lml, f, (pi, W_sr, L, b, a)


@gin.configurable
class ResNet(PredictionWrapper, hk.nets.ResNet):
  """Abstract ResNet."""

  loss = staticmethod(xent_loss)
  accuracy = staticmethod(one_hot_accuracy)

  def __init__(
    self,
    width: int = 16,
    bn_config: Optional[Mapping[str, float]] = None,
    resnet_v2: bool = False,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
  ):
    """Constructs a ResNet model.

    Args:
      width: Width multiplier.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    channels_per_group = (width, 2 * width, 4 * width, 8 * width)
    super().__init__(
      num_classes=2,
      bn_config=bn_config,
      resnet_v2=resnet_v2,
      logits_config=logits_config,
      name=name,
      initial_conv_config={"output_channels": width},
      channels_per_group=channels_per_group,
      **self.CONFIG,
    )


@gin.configurable
class ResNet10(ResNet):
  """ResNet10."""

  CONFIG = {
    "blocks_per_group": (1, 1, 1, 1),
    "bottleneck": False,
    "use_projection": (False, True, True, True),
  }


@gin.configurable
class ResNet18(ResNet):
  """ResNet18."""

  CONFIG = {
    "blocks_per_group": (2, 2, 2, 2),
    "bottleneck": False,
    "use_projection": (False, True, True, True),
  }


@gin.configurable
class ResNet34(ResNet):
  """ResNet34."""

  CONFIG = {
    "blocks_per_group": (3, 4, 6, 3),
    "bottleneck": False,
    "use_projection": (False, True, True, True),
  }


@gin.configurable
class RNN(hk.Module):
  """A multilayer LSTM"""

  loss = staticmethod(xent_loss)
  accuracy = staticmethod(one_hot_accuracy)

  def __init__(
    self,
    vocab_size: int = 81743,  # see imdb_data_augment_labels.ipynb
    embedding_size: int = 10,
    lstm_layers: Iterable[int] = [75, 30],
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    dropout: float = 0.1,
  ):
    super().__init__()
    self.embedding = hk.Embed(vocab_size, embedding_size)
    self.activation = activation
    stack = []
    for hidden_size in lstm_layers:
      stack.append(hk.LSTM(hidden_size=hidden_size))
      stack.append(self.activation)
    self.core = hk.DeepRNN(stack[:-1])
    self.dropout = dropout

  def unroll_net(self, seqs: jnp.ndarray):
    """Unrolls an LSTM over seqs, mapping each output to a scalar."""
    # seqs is [Batch, Time, Feature].
    batch_size = seqs.shape[0]
    outs, state = hk.dynamic_unroll(
      self.core, seqs, self.core.initial_state(batch_size), time_major=False
    )
    return outs, state

  def __call__(self, x, is_training):
    del is_training
    x = self.embedding(x.astype(int))
    x, _ = self.unroll_net(x)
    # x = hk.dropout(rate = self.dropout)
    B, T, F = x.shape
    x = hk.Flatten()(x)
    x = hk.Linear(2)(x)
    return Prediction(x)


@gin.register
class ViTPatchesConfig(NamedTuple):
  size: int


@gin.register
class ViTTransformerConfig(NamedTuple):
  mlp_dim: int
  num_heads: int
  num_layers: int
  attention_dropout_rate: float
  dropout_rate: float


@gin.configurable
class VisionTransformer(vit_jax_models.VisionTransformer):
  """Abstract ResNet."""

  loss = staticmethod(xent_loss)
  accuracy = staticmethod(one_hot_accuracy)

  def __call__(self, state, batch, is_training):
    del state
    x = batch.examples

    try:
      output = Prediction(super().__call__(inputs=x, train=is_training))
    except TypeError as e:
      if "train" in str(e):
        output = Prediction(super().__call__(inputs=x))
      else:
        raise e

    return output, None  # None for state.
