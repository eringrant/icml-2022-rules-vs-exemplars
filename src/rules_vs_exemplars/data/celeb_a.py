"""Biased exposure variant of the CelebA dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import os

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_datasets.image.celeba import (
  CelebA,
  IMG_ALIGNED_DATA as _CELEB_A_IMG_ALIGNED_DATA,
  EVAL_LIST as _CELEB_A_EVAL_LIST,
  ATTR_DATA as _CELEB_A_ATTR_DATA,
  ATTR_HEADINGS as _CELEB_A_ATTR_HEADINGS,
)

from rules_vs_exemplars.data.base import BiasedExposureConfig
from rules_vs_exemplars.data.base import BiasedExposureDataset
from rules_vs_exemplars.data.base import DatasetSplit


_CELEB_A_IMG_ALIGNED_DATA_FOLDER = "img_align_celeba.zip"
_CELEB_A_IMAGE_LIST_FILENAME = "list_eval_partition.txt"
_CELEB_A_ATTR_DATA_FILENAME = "list_attr_celeba.txt"


_CELEB_A_IMAGE_SHAPE = (218, 178, 3)


# We need to ensure that attributes have joint support in the dataset.
# Combinations were found with `preprocess_and_visualize_celeb_a.py`.
_CELEB_A_VALID_ATTRIBUTE_LABELS = [
  "Wearing_Lipstick",
  "Mouth_Slightly_Open",
  "Male",
  "High_Cheekbones",
  "Arched_Eyebrows",
  "Blond_Hair",
]
_CELEB_A_VALID_ATTRIBUTES = [
  _CELEB_A_ATTR_HEADINGS.index(label)
  for label in _CELEB_A_VALID_ATTRIBUTE_LABELS
]
_CELEB_A_VALID_ATTRIBUTE_COMBOS = np.array(
  list(itertools.permutations(_CELEB_A_VALID_ATTRIBUTES, 2))
)

_CELEB_A_VALID_RHO_PAIRS = [
  (0.0, 0.0),  # "zero shot"
  (1.0, 0.0),  # "cue conflict"
  (0.5, 0.0),  # "partial exposure"
  (0.5, 0.5),  # full exposure
  ## interpolation from partial exposure to zero shot:
  (0.32, 0.0),
  (0.125, 0.0),
  ## interpolation from partial exposure to full exposure:
  (0.5, 0.1),
  (0.5, 0.25),
  ## equalized spurious correlation to partial exposure
  (0.66, 0.1),
  (0.825, 0.25),
]

# Alternate approach: Subsample attributes.
# Since CelebA has 40 attributes, there are 1560 possible discriminator-
# distractor pairs; we subsample attributes here to reduce the combinatorial
# complexity.
# _NUM_ATTRIBUTES_TO_COMBINE = 5
# _SUBSAMPLED_CELEB_A_ATTRIBUTE_COMBOS = np.array(
#   list(itertools.permutations(np.arange(_NUM_ATTRIBUTES_TO_COMBINE), 2))
# )

# Alternate approach: Subsample datasets instead of attributes.
# _CELEB_A_ATTRIBUTE_COMBOS = np.array(list(itertools.permutations(
#                                      np.arange(_CELEB_A_NUM_ATTRIBUTES), 2)))
# _NUM_DATASETS_TO_GENERATE = 10
# _SUBSAMPLED_CELEB_A_ATTRIBUTE_COMBOS = _CELEB_A_ATTRIBUTE_COMBOS[
# np.random.choice(len(_CELEB_A_ATTRIBUTE_COMBOS),
#                  size=_NUM_DATASETS_TO_GENERATE)]


@gin.configurable
class BiasedExposureCelebA(BiasedExposureDataset):

  VERSION = tfds.core.Version("1.0.0")
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    From http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, download
    and place {}, {}, and {} in the manual folder (defaults to
    `~/tensorflow_datasets/downloads/manual/`).
  """.format(
    _CELEB_A_IMG_ALIGNED_DATA_FOLDER,
    _CELEB_A_IMAGE_LIST_FILENAME,
    _CELEB_A_ATTR_DATA_FILENAME,
  )

  BUILDER_CONFIGS = [
    BiasedExposureConfig(
      name="CelebA-rho0_{}-rho1_{}-discriminator_{}-distractor_{}".format(
        rho0,
        rho1,
        _CELEB_A_ATTR_HEADINGS[disc],
        _CELEB_A_ATTR_HEADINGS[dist],
      ),
      version="0.0.1",
      description=_DESCRIPTION,
      rho0=rho0,
      rho1=rho1,
      discriminator=disc,
      distractor=dist,
    )
    for ((rho0, rho1), (disc, dist),) in itertools.product(
      _CELEB_A_VALID_RHO_PAIRS,
      _CELEB_A_VALID_ATTRIBUTE_COMBOS,
    )
  ]

  MAX_EXAMPLES_PER_CONDITION = 20000

  @property
  def image_shape(self):
    return _CELEB_A_IMAGE_SHAPE

  @property
  def citation(self):
    return _CITATION

  @property
  def attribute_combos(self):
    return _CELEB_A_VALID_ATTRIBUTE_COMBOS

  @property
  def attributes(self):
    return _CELEB_A_ATTR_HEADINGS

  @property
  def num_attributes(self):
    return len(self.attributes)

  def _extract_celeba(img_list_path, attr_path):

    with tf.io.gfile.GFile(img_list_path) as f:
      train_files, valid_files, test_files = [], [], []
      for line in f.readlines():
        img = line.split()[0]
        if int(line.split()[1]) == 0:
          train_files += [img]
        elif int(line.split()[1]) == 1:
          valid_files += [img]
        elif int(line.split()[1]) == 2:
          test_files += [img]
        else:
          raise ValueError("Invalid split specification in CelebA partition.")

    _, attributes = CelebA._process_celeba_config_file(None, attr_path)
    return train_files, valid_files, test_files, attributes

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""

    try:
      extracted_dirs = dl_manager.download_and_extract(
        {
          "img_align_celeba": _CELEB_A_IMG_ALIGNED_DATA,
          "list_eval_partition": _CELEB_A_EVAL_LIST,
          "list_attr_celeba": _CELEB_A_ATTR_DATA,
        }
      )
      (
        train_files,
        valid_files,
        test_files,
        attributes,
      ) = BiasedExposureCelebA._extract_celeba(
        img_list_path=extracted_dirs["list_eval_partition"],
        attr_path=extracted_dirs["list_attr_celeba"],
      )
      image_path = os.path.join(
        extracted_dirs["img_align_celeba"],
        "img_align_celeba",
      )

    except Exception:
      # Likely an error in downloading the files; follow the instructions in
      # `BiasedExposureCelebA.MANUAL_DOWNLOAD_INSTRUCTIONS`.
      extracted_dirs = {}
      (
        train_files,
        valid_files,
        test_files,
        attributes,
      ) = BiasedExposureCelebA._extract_celeba(
        img_list_path=os.path.join(
          dl_manager.manual_dir, _CELEB_A_IMAGE_LIST_FILENAME
        ),
        attr_path=os.path.join(
          dl_manager.manual_dir, _CELEB_A_ATTR_DATA_FILENAME
        ),
      )
      image_path = os.path.join(
        dl_manager.extract(
          os.path.join(dl_manager.manual_dir, _CELEB_A_IMG_ALIGNED_DATA_FOLDER)
        ),
        "img_align_celeba",
      )

    assert set(train_files).isdisjoint(set(valid_files))
    assert set(valid_files).isdisjoint(set(test_files))
    assert set(train_files).isdisjoint(set(test_files))

    train_attributes = np.vstack(
      attributes[train_file] for train_file in train_files
    )
    valid_attributes = np.vstack(
      attributes[valid_file] for valid_file in valid_files
    )
    test_attributes = np.vstack(
      attributes[test_file] for test_file in test_files
    )

    train_attributes = train_attributes > 0
    valid_attributes = valid_attributes > 0
    test_attributes = test_attributes > 0

    train_files = (
      os.path.join(image_path, train_file) for train_file in train_files
    )
    valid_files = (
      os.path.join(image_path, valid_file) for valid_file in valid_files
    )
    test_files = (
      os.path.join(image_path, test_file) for test_file in test_files
    )

    # Specify the splits
    return {
      "train": self._generate_examples(
        examples=train_files,
        attributes=train_attributes,
        split=DatasetSplit.TRAIN,
      ),
      "valid": self._generate_examples(
        examples=valid_files,
        attributes=valid_attributes,
        split=DatasetSplit.TRAIN,
      ),
      "test": self._generate_examples(
        examples=test_files,
        attributes=test_attributes,
        split=DatasetSplit.EVAL,
      ),
    }


if __name__ == "__main__":
  for cfg in BiasedExposureCelebA.BUILDER_CONFIGS:
    dataset = BiasedExposureCelebA(config=cfg)
    dataset.download_and_prepare()
