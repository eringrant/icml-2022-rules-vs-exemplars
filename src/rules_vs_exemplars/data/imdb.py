"""Generates data for imbd exposure bias experiments."""

import numpy as np
import pandas as pd
import os

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from rules_vs_exemplars.data.base import Batch
from string import punctuation
from collections import defaultdict


_IMDB_VALID_ATTRIBUTES = ["an", "are", "at", "film", "one", "you"]

_IMDB_VALID_ATTRIBUTES_LABELS = [
  "has_" + word for word in _IMDB_VALID_ATTRIBUTES
]


# where augmented data lives
PATH = "~/tensorflow_datasets/imdb_reviews/plain_text/"
MAX_SENT_LEN = 200


def word_to_float_mapping(reviews, vocab_to_float):
  """
      Transforms String review in dataframe into int encoded review.

  :param reviews: pd.Series of reviews to transform
  :param vocab_to_float: Word to int mapping as a dictionary
  :return: Transformed Dataframe
  """
  reviews_int = []
  for review in reviews:
    r = []
    for w in review.split():
      r.append(vocab_to_float[w])
    reviews_int.append(r)

  features = pad_features(reviews_int, MAX_SENT_LEN).reshape(-1, MAX_SENT_LEN)
  return features


def pad_features(reviews_int, seq_length):
  """
      Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.

  :param reviews_int: Reviews as array of ints
  :param seq_length: Length of Sequence for the review
  :return: 0 padded review
  """
  features = np.zeros((len(reviews_int), seq_length), dtype=float)

  for i, review in enumerate(reviews_int):
    review_len = len(review)

    if review_len <= seq_length:
      zeroes = list(np.zeros(seq_length - review_len))
      new = zeroes + review
    elif review_len > seq_length:
      new = review[0:seq_length]

    features[i, :] = np.array(new)

  return features


def preprocess(reviews, vocab_to_float_dict):
  """
      Preproceseds text data into int encoded.

  :param reviews: pd.Series to transform
  :return: Transformed pd.Series
  """

  # standardize
  reviews = reviews.apply(lambda x: x.lower())
  reviews = reviews.apply(
    lambda x: "".join([c for c in x if c not in punctuation])
  )

  return word_to_float_mapping(reviews, vocab_to_float_dict)


@gin.configurable
class IMDBDataset(object):
  """
  Generates a data object.
  """

  def __new__(
    self,
    expt: str,
    split: str,
    distractor_name: str,
    test_distractor_label_value: int,
    test_discriminator_label_value: int,
    numdata: int,
  ):

    inf_label = "label"
    if distractor_name not in _IMDB_VALID_ATTRIBUTES_LABELS:
      raise ValueError("Invalid distractor label")
    if test_discriminator_label_value not in [0, 1]:
      raise ValueError("Invalid test discriminator label")
    if test_distractor_label_value not in [0, 1]:
      raise ValueError("Invalid test distractor label")

    # get data
    DATAFRAME = pd.read_csv(PATH + "with_additional_labels.zip", nrows=numdata)
    # get vocabulary
    VOCAB = pd.read_csv(PATH + "wordcounts.zip")

    # make vocab to int dictionary
    # defaults all unseen words (that appear exactly once in data) to 0.0
    VOCAB_TO_FLOAT = defaultdict(float)
    for i, w in enumerate(VOCAB["word"]):
      VOCAB_TO_FLOAT[w] = i + 1.0

    true_label = DATAFRAME["label"]
    true_label_onehot = np.zeros((true_label.shape[0], 2))
    true_label_onehot[np.arange(true_label.shape[0]), true_label] = 1

    distractor_name = DATAFRAME[distractor_name]

    data = DATAFRAME["text"]
    data = preprocess(data, vocab_to_float_dict=VOCAB_TO_FLOAT)

    # get minimum number per quadrant
    # distractors generated to match within 5% on sample data
    # but not exactly matched.
    # min_N = np.inf
    # for disc_val in [0,1]:
    # for dist_val in [0,1]:
    # overlap = np.sum(
    # np.logical_and(
    # (distractor_name == dist_val),
    # (discriminator_label == disc_val)
    # )
    # )
    # min_N = min(min_N, overlap)

    # denote labels by whether they are same as test set or not
    true_label_eq2test = true_label == test_discriminator_label_value
    distractor_label_eq2test = distractor_name == test_distractor_label_value

    mask = np.logical_and(true_label_eq2test, distractor_label_eq2test)

    # assign test data
    test_x, test_y = data[mask], true_label_onehot[mask]

    # the train set depends on the experiment
    if expt == "zero_shot":
      mask = np.logical_not(distractor_label_eq2test)

    elif expt == "conflict":
      mask = np.logical_xor(true_label_eq2test, distractor_label_eq2test)

    elif expt == "partial_exposure":
      balance = True
      if balance:
        # balance data across informative dimension

        # include all data with same true label as test
        mask1 = np.logical_and(
          true_label_eq2test, np.logical_not(distractor_label_eq2test)
        )

        # include half of others
        mask2 = np.logical_and(
          np.logical_not(true_label_eq2test),
          np.random.uniform(size=true_label.shape[0]) > 0.5,
        )

        mask = np.logical_or(mask1, mask2)

      else:
        mask = np.logical_not(
          np.logical_and(true_label_eq2test, distractor_label_eq2test)
        )

    else:
      raise ValueError("Enter valid experiment")

    # Subsample so that train and valid are always slightly different
    subsample = np.random.uniform(size=mask.shape) > 0.1
    mask = np.logical_and(mask, subsample)

    train_x, train_y = data[mask], true_label_onehot[mask]
    
    # dist = 1, disc = 1, is the default test quadrant
    
    train_disc, train_dist = (train_y == test_discriminator_label_value, 
                              train_y == test_distractor_label_value)
    test_disc, test_dist = (test_y == test_discriminator_label_value, 
                            test_y == test_distractor_label_value)

    ds = {
      "train": Batch(train_x, train_y, train_disc, train_dist),
      "valid": Batch(train_x, train_y, train_disc, train_dist),
      "test": Batch(test_x, test_y, test_disc, test_dist),
    }

    return tf.data.Dataset.from_tensor_slices(ds[split])
