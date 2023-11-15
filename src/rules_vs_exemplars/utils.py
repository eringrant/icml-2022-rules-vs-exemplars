import contextlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from absl import logging



@contextlib.contextmanager
def time_activity(activity_name: str):
  logging.info("[Timing] %s start.", activity_name)
  yield
  logging.info("[Timing] %s finished.", activity_name)


def visualize_decision_boundary(
  name, forward_pass_fn, train_plot_data, test_plot_data
):
  
  # rotate plot data by 90 degrees for visual similarity
  # just changed the test labels in models.py, remove rotation
  rotation_matrix = np.array([np.array([1, 0]), 
                             np.array([0, 1])])

  rotated_train = np.matmul(train_plot_data.examples, 
                                       rotation_matrix)
  rotated_test = np.matmul(test_plot_data.examples, 
                                       rotation_matrix)
  train_targets = train_plot_data.targets

  xx, yy = np.mgrid[-15:15:0.9, -15:15:0.9]
  grid_data = np.c_[xx.ravel(), yy.ravel()]
  
  grid_data = np.matmul(grid_data, rotation_matrix)

  logits = forward_pass_fn(grid_data)
  probs = softmax(logits, axis=1)[:, 0].reshape(xx.shape)

  plt.contourf(xx, yy, probs, 25, cmap="PuOr", vmin=0, vmax=1)

  train_feps = rotated_train[np.argmax(train_targets, axis = 1) == 1]
  plt.scatter(
    train_feps[:, 0],
    train_feps[:, 1],
    c="tab:orange",
    marker="o",
    label="train",
  )
  train_zups = rotated_train[np.argmax(train_targets, axis = 1) == 0]
  plt.scatter(
    train_zups[:, 0],
    train_zups[:, 1],
    c="tab:purple",
    marker="o",
    label="train",
  )
  
  plt.scatter(
    rotated_test[:, 0],
    rotated_test[:, 1],
    c="tab:orange",
    marker="^",
    label="test",
  )  

  plt.legend()

  plt.title(name)
  if name:
    plt.savefig(name + ".png")
    
  plt.close()

