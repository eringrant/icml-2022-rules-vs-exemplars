from absl import logging
from absl import flags
import os
import pandas as pd

from typing import Any
from typing import Generator
from typing import Tuple
from typing import Callable
from typing import Union

from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import scipy
import tree

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from rules_vs_exemplars import data
from rules_vs_exemplars import models
from rules_vs_exemplars import utils


import flax.linen as nn
import vit_jax
from vit_jax import checkpoint

gin.external_configurable(optax.adam)

RANDOM_SEED = 42

OptState = Any


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


def forward(
  model: hk.Module,
  batch: data.Batch,
  is_training: bool,
  **kwargs,
) -> jnp.ndarray:
  """Forward application of the model."""
  return model()(batch.examples, is_training=is_training, **kwargs)


def weight_decay_params(params):
  def is_not_bias_or_bn_param(module_name, param_name):
    return not (param_name == "b" or "batchnorm" in module_name)

  return [
    p
    for ((module_name, param_name), p) in tree.flatten_with_path(params)
    if is_not_bias_or_bn_param(module_name, param_name)
  ]


@gin.configurable
def objective(
  params: hk.Params,
  state: hk.State,
  forward_fn: Callable,
  loss_fn: Callable,
  batch: data.Batch,
  regularization_coefficient: float,
  is_training: bool,
) -> Tuple[jnp.ndarray]:
  """Computes a regularized loss for the given batch."""
  predictions, state = forward_fn.apply(
    params,
    state,
    batch=batch,
    is_training=is_training,
  )
  cat_loss = jnp.mean(
    loss_fn(predictions=predictions, onehot_labels=batch.targets)
  )

  if regularization_coefficient > 0:
    reg_loss = regularization_coefficient * models.l2_loss(
      weight_decay_params(params)
    )
  else:
    reg_loss = 0.0

  return cat_loss + reg_loss, state


def accuracy(
  params: hk.Params,
  state: hk.State,
  forward_fn: Callable,
  batch: data.Batch,
  is_training: bool,
) -> Tuple[jnp.ndarray]:
  """Computes a regularized loss for the given batch."""
  predictions, _ = forward_fn.apply(
    params,
    state,
    batch=batch,
    is_training=is_training,
  )
  return jnp.argmax(predictions.logits, axis=-1) == jnp.argmax(
    batch.targets, axis=-1
  )


def update(
  params: hk.Params,
  state: hk.State,
  objective_fn: Callable,
  opt_state: OptState,
  batch: data.Batch,
  opt: Callable,
) -> Tuple[hk.Params, OptState]:
  """Learning rule (stochastic gradient descent)."""
  (loss, state), grads = jax.value_and_grad(objective_fn, has_aux=True)(
    params,
    state,
    batch=batch,
    is_training=True,
  )
  updates, opt_state = opt.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return params, state, opt_state


def random_accuracy_fn(onehot_targets: jnp.array):
  targets = jnp.argmax(onehot_targets, axis=-1)
  mode = scipy.stats.mode(targets).mode
  return jnp.mean(targets == mode)


def casewise_accuracy(corrects, batch):
  disc = jnp.argmax(batch.discriminator, axis=-1)
  dist = jnp.argmax(batch.distractor, axis=-1)

  off_off = jnp.logical_and(jnp.logical_not(disc), jnp.logical_not(dist))
  on_off = jnp.logical_and(disc, jnp.logical_not(dist))
  off_on = jnp.logical_and(jnp.logical_not(disc), dist)
  on_on = jnp.logical_and(disc, dist)

  return (
    corrects[off_off].mean(),
    corrects[on_off].mean(),
    corrects[off_on].mean(),
    corrects[on_on].mean(),
  )


def get_forward_fn(model: Union[hk.Module, nn.Module]):
  # Construct forward pass, etc. functions for `haiku`.
  if issubclass(model, hk.Module):
    forward_fn = hk.without_apply_rng(
      hk.transform_with_state(partial(forward, model=model))
    )

  # Hacked-in compatibility for `flax.linen` models.
  elif issubclass(model, nn.Module):
    module = model()
    forward_fn = module

  else:
    raise NotImplementedError(f"Model class {model} unknown.")

  return forward_fn


@gin.configurable(
  allowlist=(
    "opt",
    "model",
    "max_num_steps",
    "validation_interval",
    "random_seed",
    "visualize",
    "evaluate_on_test",
    "early_stopping_tolerance",
    "early_stopping_min_steps",
    "early_stopping_patience",
  )
)
def train(
  train_dataset: tf.data.Dataset,
  valid_dataset: tf.data.Dataset,
  model: Union[hk.Module, nn.Module],
  opt: Callable,
  max_num_steps: int,
  validation_interval: int,
  random_seed: int = 42,
  visualize: bool = False,
  evaluate_on_test: bool = False,
  test_dataset: tf.data.Dataset = None,
  early_stopping_tolerance: float = 0.01,
  early_stopping_min_steps: int = 500,
  early_stopping_patience: int = 50,
) -> Tuple[hk.Params, hk.State, OptState]:

  assert (
    early_stopping_patience >= validation_interval
  ), "Should increase patience least as much as the validation interval."
  assert early_stopping_min_steps % validation_interval == 0, (
    "Should validate interval at the first iteration that early stopping can "
    "take place."
  )

  forward_fn = get_forward_fn(model)

  batch = next(train_dataset)
  try:
    params, state = forward_fn.init(
      jax.random.PRNGKey(random_seed),
      batch=batch,
      is_training=True,
    )
  except TypeError:  # need to use dummy state
    state = None
    params = forward_fn.init(
      jax.random.PRNGKey(random_seed),
      state=state,
      batch=batch,
      is_training=True,
    )

  opt_state = opt.init(params)

  objective_fn = partial(objective, forward_fn=forward_fn, loss_fn=model.loss)
  accuracy_fn = partial(accuracy, forward_fn=forward_fn)

  logging.info(
    "Operative Gin configuration for training:\n%s", gin.operative_config_str()
  )

  if visualize:
    lc_saveto = flags.FLAGS.summary_dir + "_lcurve.txt"
    with open(lc_saveto, "a") as f:
      f.write(
        "Train_acc, Valid_acc, Test_acc, Train_loss, Valid_loss, Test_loss \n"
      )

    plot_saveto = flags.FLAGS.summary_dir + "_plots/"
    if not os.path.exists(plot_saveto):
      os.makedirs(plot_saveto)

    # Construct forward pass specifically for the visualizer.
    def forward_pass_for_visualizer(batch, params, state):
      batch = data.Batch(batch, None)
      return forward_fn.apply(params, state, batch=batch, is_training=False)[
        0
      ].logits

  # Training and evaluation loop.
  best_valid_loss = jnp.inf
  best_params = params
  best_state = state
  for step in range(max_num_steps):

    train_batch = next(train_dataset)

    if step % validation_interval == 0:

      # Evaluation on the training set.
      train_loss, _ = objective_fn(
        params=params,
        state=state,
        batch=train_batch,
        is_training=True,
      )
      train_accuracy = accuracy_fn(
        params=params,
        state=state,
        batch=train_batch,
        is_training=True,
      )
      rand_train_accuracy = random_accuracy_fn(train_batch.targets)

      # Evaluation on the validation set.
      valid_batch = next(valid_dataset)
      valid_loss, _ = objective_fn(
        params=params,
        state=state,
        batch=valid_batch,
        is_training=False,
      )
      valid_accuracy = accuracy_fn(
        params=params,
        state=state,
        batch=valid_batch,
        is_training=False,
      )
      rand_valid_accuracy = random_accuracy_fn(valid_batch.targets)

      # Evaluation on the testing set (for learning curves on toy datasets).
      if evaluate_on_test:
        test_batch = next(test_dataset)
        test_accuracy = accuracy_fn(
          params=params,
          state=state,
          batch=test_batch,
          is_training=False,
        )
        test_loss, _ = objective_fn(
          params=params,
          state=state,
          batch=test_batch,
          is_training=False,
        )
        rand_test_accuracy = random_accuracy_fn(test_batch.targets)

      train_casewise_accuracy = casewise_accuracy(train_accuracy, train_batch)
      valid_casewise_accuracy = casewise_accuracy(valid_accuracy, valid_batch)

      logging.info(
        f"[step {step:05}]\t"
        f"train: loss {train_loss:.3f}  "
        f"accuracy {jnp.mean(train_accuracy) * 100:.1f}%  "
        f"(by case {train_casewise_accuracy[0] * 100:.1f}%"
        f" {train_casewise_accuracy[1] * 100:.1f}%"
        f" {train_casewise_accuracy[2] * 100:.1f}%"
        f" {train_casewise_accuracy[3] * 100:.1f}%)  "
        f"random {rand_train_accuracy * 100:.1f}%  |  "
        f"valid: loss {valid_loss:.3f}  "
        f"accuracy {jnp.mean(valid_accuracy) * 100:.1f}%  "
        f"(by case {valid_casewise_accuracy[0] * 100:.1f}%"
        f" {valid_casewise_accuracy[1] * 100:.1f}%"
        f" {valid_casewise_accuracy[2] * 100:.1f}%"
        f" {valid_casewise_accuracy[3] * 100:.1f}%)  "
        f"random {rand_valid_accuracy * 100:.1f}%"
        + (
          f"  |  test: loss {test_loss:.3f} "
          f"accuracy {jnp.mean(test_accuracy) * 100:.1f}%  "
          f"random {rand_test_accuracy * 100:.1f}%"
          if evaluate_on_test
          else ""
        )
      )

      if step > early_stopping_min_steps:

        # Early stopping on the validation set.
        if valid_loss > (best_valid_loss * (1.0 + early_stopping_tolerance)):
          logging.info("Validation loss did not improve; stopping early.")
          break

        if valid_loss < best_valid_loss:
          logging.info(
            "Increasing patience due to a strict improvement in the "
            "validation loss."
          )
          early_stopping_min_steps += early_stopping_patience
          best_valid_loss = valid_loss
          best_params = params
          best_state = state

      # Visualization for toy datasets.
      if visualize:
        utils.visualize_decision_boundary(
          plot_saveto + str(step),
          partial(forward_pass_for_visualizer, params=params, state=state),
          train_plot_data=train_batch,
          test_plot_data=test_batch,
        )

        with open(lc_saveto, "a") as f:
          f.write(
            f"{train_accuracy:.3f}, {valid_accuracy:.3f}, "
            f"{test_accuracy:.3f}, {train_loss:.3f}, "
            f"{valid_loss:.3f}, {test_loss:.3f} \n"
          )

      # Explicitly garbage-collect, because Python doesn't do so.
      del valid_batch
      if evaluate_on_test:
        del test_batch

    # Do SGD on a batch of training examples.
    params, state, opt_state = update(
      objective_fn=objective_fn,
      params=params,
      state=state,
      batch=train_batch,
      opt_state=opt_state,
      opt=opt,
    )

    # Explicitly garbage-collect, because Python doesn't do so.
    del train_batch

  return best_params, best_state


@gin.configurable(allowlist=("model",))
def evaluate(
  params: hk.Params,
  state: hk.State,
  test_dataset: tf.data.Dataset,
  model: hk.Module,
  visualize: bool = False,
):

  forward_fn = get_forward_fn(model)

  def evaluate_batch(batch: data.Batch) -> jnp.ndarray:
    predictions, _ = forward_fn.apply(
      params, state, batch=batch, is_training=False
    )
    predicted_label = jnp.argmax(predictions.logits, axis=-1)
    correct = jnp.equal(predicted_label, jnp.argmax(batch.targets, axis=-1))
    return correct.astype(jnp.float32)

  logging.info(
    "Operative Gin configurations for evaluation:\n%s",
    gin.operative_config_str(),
  )

  results = []
  for test_batch in test_dataset:
    results += list(
      zip(
        evaluate_batch(test_batch),
        tf.argmax(test_batch.discriminator, axis=-1),
        tf.argmax(test_batch.distractor, axis=-1),
        # test_batch.examples * 255,
      )
    )
    del test_batch

  results = pd.DataFrame(
    data=results,
    columns=["correct", "discriminator", "distractor"],  # 'image'],
    dtype=int,
  )

  logging.info(results.describe())
  test_savepath = os.path.join(flags.FLAGS.summary_dir, "test_results.csv")
  results.to_csv(test_savepath)
  logging.info(f"Wrote test results to {test_savepath}.")


@gin.configurable(
  allowlist=(
    "dataset_name",
    "train_batch_size",
    "valid_batch_size",
    "test_batch_size",
  )
)
def load_datasets(
  dataset_name: str,
  train_batch_size: int,
  valid_batch_size: int,
  test_batch_size: int,
) -> Generator[data.Batch, None, None]:
  """Loads the datasets as generators of batches."""
  # Generalize if using other image datasets.
  if "biased_exposure_celeb_a" in dataset_name:
    dataset = tfds.load(dataset_name)
    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    def wrap_named_tuple(d):
      # Sanity-check normalization scheme.
      return data.Batch(
        tf.one_hot(d["discriminator"], 2),
        tf.one_hot(d["discriminator"], 2),
        tf.one_hot(d["distractor"], 2),
      )

    train_dataset = train_dataset.map(wrap_named_tuple)
    valid_dataset = valid_dataset.map(wrap_named_tuple)
    test_dataset = test_dataset.map(wrap_named_tuple)

  elif dataset_name == "linear":
    train_dataset = data.LinearDataset(split="train")
    valid_dataset = data.LinearDataset(split="valid")
    test_dataset = data.LinearDataset(split="test")

  elif dataset_name == "imdb":
    train_dataset = data.IMDBDataset(split="train")
    valid_dataset = data.IMDBDataset(split="valid")
    test_dataset = data.IMDBDataset(split="test")

  else:
    raise ValueError("Unrecognized dataset.")

  # CelebA does not need shuffling.
  # train_dataset = train_dataset.shuffle(10 * train_batch_size, seed=0)
  train_dataset = train_dataset.batch(train_batch_size)
  train_dataset = train_dataset.repeat()

  valid_dataset = valid_dataset.repeat()
  valid_dataset = valid_dataset.batch(valid_batch_size)

  test_dataset = test_dataset.batch(test_batch_size)

  return (
    iter(tfds.as_numpy(train_dataset)),
    iter(tfds.as_numpy(valid_dataset)),
    iter(tfds.as_numpy(test_dataset)),
  )
