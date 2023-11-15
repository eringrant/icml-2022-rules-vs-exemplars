"""Generates data for linear exposure bias experiments."""

import numpy as np
import random

import gin
import tensorflow as tf

from rules_vs_exemplars.data.base import Batch


@gin.configurable
class LinearDataset(object):
  """
  Generates a data object.
  """
  SEED = 123

  def __new__(
    self,
    expt: str,
    split: str,
    x_offset_abs: float,
    y_offset_abs: float,
    num_datapoints: int,
    center_value: float = 0.0,
  ):
    
    # TempHack: fixing random seed to prevent different randomization 
    # across train/valid/test calls
    # randomization only needed in subsampling for partial_exposure
    np.random.seed(seed=self.SEED)    

    inf_dim = 1
    distractor_dim = 0

    test_true_label = 0
    test_distractor_label = 1

    # generate data
    data_list = []
    for x in [-x_offset_abs, x_offset_abs]:
      for y in [-y_offset_abs, y_offset_abs]:
        center = center_value + np.array([x, y])[None, :]
        data_list.append(
          center
          + np.reshape(np.random.normal(size=num_datapoints * 2), (-1, 2))
        )

    data = np.concatenate(data_list, axis=0)

    # classify the data
    true_label = (data[:, inf_dim] > center_value).astype("int")

    true_label_onehot = np.zeros((4 * num_datapoints, 2))
    true_label_onehot[np.arange(4 * num_datapoints), true_label] = 1

    # track confounder values
    distractor_label = (data[:, distractor_dim] > center_value).astype("int")

    # denote labels by whether they are same as test set or not
    true_label_eq2test = true_label == test_true_label
    distractor_label_eq2test = distractor_label == test_distractor_label

    mask = np.logical_and(true_label_eq2test, distractor_label_eq2test)
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
          np.random.uniform(size=4 * num_datapoints) > 0.5,
        )

        mask = np.logical_or(mask1, mask2)

      else:
        mask = np.logical_not(
          np.logical_and(true_label_eq2test, distractor_label_eq2test)
        )

    else:
      raise ValueError("Enter valid experiment")

    train_x0, train_y0 = data[mask], true_label_onehot[mask]
    
    shuffle = np.random.permutation(len(train_y0))
    
    train_x, train_y = train_x0[shuffle], train_y0[shuffle]

    ds = {
      "train": Batch(train_x, train_y),
      "valid": Batch(train_x, train_y),
      "test": Batch(test_x, test_y),
    }
    return tf.data.Dataset.from_tensor_slices(ds[split])
