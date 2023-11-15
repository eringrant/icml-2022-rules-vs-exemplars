"""Abstract base classes for implementing biased exposure datasets.

See `biased_exposure/data/celeb_a.py` for an example implemented subclass.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataclasses import dataclass
from enum import auto, Enum, unique
import itertools
import numpy as np
from typing import Iterable
from typing import NamedTuple

import networkx as nx
import tensorflow_datasets as tfds


class Batch(NamedTuple):
  examples: np.ndarray
  targets: np.ndarray
  discriminator: np.ndarray
  distractor: np.ndarray


def threshold_attribute_combos(attr: np.ndarray, t: int) -> Iterable[set]:
  """Thresholds the attribute pairs in `attr` by `t`.

  Returns a maximal list of attributes of which all pairs of values have
  joint support in at least `t` examples, as described by the (example,
  attribute) matrix `attr`.
  """
  m = attr.shape[1]

  # Compute the support of all attribute value combinations.
  x_y_support_counts = np.zeros((m, m), int)
  not_x_not_y_support_counts = np.zeros((m, m), int)
  x_not_y_support_counts = np.zeros((m, m), int)
  not_x_y_support_counts = np.zeros((m, m), int)

  for (i, j) in itertools.combinations(range(m), 2):
    x_y_support_counts[i, j] = np.logical_and(attr[:, i], attr[:, j]).sum()
    x_not_y_support_counts[i, j] = np.logical_and(
      attr[:, i], np.logical_not(attr[:, j])
    ).sum()
    not_x_y_support_counts[i, j] = np.logical_and(
      np.logical_not(attr[:, i]), attr[:, j]
    ).sum()
    not_x_not_y_support_counts[i, j] = np.logical_and(
      np.logical_not(attr[:, i]), np.logical_not(attr[:, j])
    ).sum()

  # Threshold the combinations.
  valid_combos = np.where(
    np.logical_and.reduce(
      (
        x_y_support_counts > t,
        not_x_not_y_support_counts > t,
        x_not_y_support_counts > t,
        not_x_y_support_counts > t,
      )
    )
  )

  # The problem is equivalent to returning a maximal clique.
  G = nx.Graph()
  G.add_edges_from(list(zip(*valid_combos)))
  cliques = list(map(set, nx.clique.find_cliques(G)))
  return cliques


class AutoName(Enum):
  @staticmethod
  def _generate_next_value_(name, start, count, last_values):
    del start
    del count
    del last_values
    return name.lower()


@unique
class DatasetSplit(AutoName):
  TRAIN = auto()
  EVAL = auto()


def case_masking(mask, discriminator_features, distractor_features):
  case_masks = [[], []]
  for disc in [0, 1]:
    for dist in [0, 1]:
      case_mask = np.logical_and.reduce(
        [
          mask,
          discriminator_features == bool(disc),
          distractor_features == bool(dist),
        ]
      )
      case_masks[disc] += [case_mask]
  return np.array(case_masks)


def compute_case_counts(case_masks):
  case_counts = np.zeros(shape=(2, 2))
  for disc in [0, 1]:
    for dist in [0, 1]:
      case_counts[disc, dist] = sum(case_masks[disc][dist])
  return case_counts


def subsample_mask(mask, p):
  """Subsample `mask` by proportion `p`."""
  assert 0.0 <= p <= 1.0
  return np.logical_and(mask, np.random.uniform(size=mask.shape[0]) < p)


def balance_masks(mask0, mask1):
  """Balance magnitudes of `mask0` and `mask1` by subsampling the largest."""
  N0, N1 = sum(mask0), sum(mask1)
  if N0 > N1:
    return subsample_mask(mask0, N1 / N0), mask1
  elif N0 < N1:
    return mask0, subsample_mask(mask1, N0 / N1)
  else:
    # Already balanced.
    return mask0, mask1


def cap_mask(mask, N):
  """Cap the magnitude of `mask` to `N` by subsampling."""
  if sum(mask) > N:
    return subsample_mask(mask, N / sum(mask))
  else:
    return mask


def rho_interpolation(features, rho):
  features0, features1 = balance_masks(features == 0, features == 1)
  return np.logical_or(
    subsample_mask(features0, 1 - rho), subsample_mask(features1, rho)
  )


def condition_mask(discriminator_features, distractor_features, rho0, rho1, N):

  # Masks for each of `discriminator` in {0, 1}.
  distractor_mask0 = rho_interpolation(
    distractor_features[discriminator_features == 0], rho0
  )
  distractor_mask1 = rho_interpolation(
    distractor_features[discriminator_features == 1], rho1
  )

  # Move back to the full dataset.
  mask0 = np.zeros_like(discriminator_features)
  mask0[discriminator_features == 0] = distractor_mask0
  mask1 = np.zeros_like(discriminator_features)
  mask1[discriminator_features == 1] = distractor_mask1

  # Cap and balance.
  mask0 = cap_mask(mask0, N)
  mask1 = cap_mask(mask1, N)
  mask0, mask1 = balance_masks(mask0, mask1)

  # Combine.
  mask = np.logical_or(mask0, mask1)

  # Some assertions:
  # Masks should be non-empty.
  assert sum(mask0) > 0 and sum(mask1) > 1
  # Masks should not overlap.
  assert sum(np.logical_and(mask0 == 1, mask1 == 1)) == 0

  # Check relative balance with some tolerance.
  case_masks = case_masking(mask, discriminator_features, distractor_features)
  case_counts = compute_case_counts(case_masks)
  print(rho0, rho1)
  print(case_counts)
  assert np.allclose(
    case_counts[0] / sum(case_counts[0]), [1 - rho0, rho0], atol=0.1
  )
  assert np.allclose(
    case_counts[1] / sum(case_counts[1]), [1 - rho1, rho1], atol=0.1
  )

  return mask


@dataclass
class BiasedExposureConfig(tfds.core.BuilderConfig):
  """`BuilderConfig` for a `BiasedExposureDataset`."""

  def __init__(
    self,
    rho0: float,
    rho1: float,
    discriminator: int,
    distractor: int,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.rho0 = rho0
    self.rho1 = rho1
    self.discriminator = discriminator
    self.distractor = distractor


class BiasedExposureDataset(tfds.core.GeneratorBasedBuilder):
  """A binary classification dataset with latent combinatorial structure."""

  SEED = 123

  @property
  def distractor(self):
    return self.builder_config.distractor

  @property
  def discriminator(self):
    return self.builder_config.discriminator

  @property
  def rho0(self):
    return self.builder_config.rho0

  @property
  def rho1(self):
    return self.builder_config.rho1

  @property
  def description(self):
    return self.builder_config.description

  def _info(self) -> tfds.core.DatasetInfo:

    return tfds.core.DatasetInfo(
      builder=self,
      description=self.description,
      features=tfds.features.FeaturesDict(
        {
          "image": tfds.features.Image(shape=self.image_shape),
          "discriminator": tfds.features.ClassLabel(names=("no", "yes")),
          "distractor": tfds.features.ClassLabel(names=("no", "yes")),
          "attributes": {name: bool for name in self.attributes},
        }
      ),
      citation=self.citation,
    )

  def _generate_examples(
    self, examples: Iterable, attributes: np.ndarray, split: DatasetSplit
  ):
    """Yields examples from this `BiasedExposureDataset`."""
    np.random.seed(self.SEED)

    if split == DatasetSplit.TRAIN:
      discriminator_features = attributes[:, self.discriminator]
      distractor_features = attributes[:, self.distractor]
      mask = condition_mask(
        discriminator_features,
        distractor_features,
        rho0=self.rho0,
        rho1=self.rho1,
        N=self.MAX_EXAMPLES_PER_CONDITION,
      )

    elif split == DatasetSplit.EVAL:
      # Evaluate on all examples.
      mask = np.ones(shape=attributes.shape[0], dtype=bool)

    else:
      raise ValueError("Unknown split.")

    for i, (example, example_attributes) in enumerate(
      zip(examples, attributes)
    ):
      if split == DatasetSplit.TRAIN and not mask[i]:
        # Skip training (and validation) examples outside exposure condition.
        continue

      else:
        distractor_feature = 1 if example_attributes[self.distractor] else 0
        discriminator_feature = (
          1 if example_attributes[self.discriminator] else 0
        )

        yield str(i), {
          "image": example,
          "distractor": distractor_feature,
          "discriminator": discriminator_feature,
          "attributes": dict(zip(self.attributes, example_attributes)),
        }
