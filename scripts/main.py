"""Training and evaluation on a biased-exposure dataset.

Adapted from github.com/deepmind/dm-haiku/blob/master/examples/mnist.py.
"""

from absl import app
from absl import flags
from absl import logging

import os
import sys

import gin

import tensorflow as tf

from rules_vs_exemplars import data
from rules_vs_exemplars import models
from rules_vs_exemplars import train


DEFAULT_DIR = "/tmp/rules_vs_exemplars/"


flags.DEFINE_string(
  "checkpoint_dir",
  os.path.join(DEFAULT_DIR, "checkpoints"),
  "Directory in which to store model checkpoints.",
)

flags.DEFINE_string(
  "summary_dir",
  os.path.join(DEFAULT_DIR, "summaries"),
  "Directory in which to store summaries.",
)

flags.DEFINE_multi_string(
  "gin_config", None, "List of paths to the Gin config files."
)
flags.DEFINE_multi_string(
  "gin_binding", None, "List of Gin parameter bindings."
)
FLAGS = flags.FLAGS


def operative_config_path(
  operative_config_dir, operative_config_filename="operative_config.gin"
):
  return os.path.join(operative_config_dir, operative_config_filename)


def load_operative_gin_configurations(operative_config_dir):
  """Load operative Gin configurations from the given directory."""
  gin_log_file = operative_config_path(operative_config_dir)
  with gin.unlock_config():
    gin.parse_config_file(gin_log_file)
  gin.finalize()
  logging.info("Operative Gin configurations loaded from %s.", gin_log_file)


def record_operative_gin_configurations(operative_config_dir):
  """Record operative Gin configurations in the given directory."""
  gin_log_file = operative_config_path(operative_config_dir)
  # If it exists already, rename it instead of overwriting it.
  # This just saves the previous one, not all the ones before.
  if tf.io.gfile.exists(gin_log_file):
    tf.io.gfile.rename(gin_log_file, gin_log_file + ".old", overwrite=True)
  with tf.io.gfile.GFile(gin_log_file, "w") as f:
    f.write(gin.operative_config_str())


def main(_):
  gin.parse_config_files_and_bindings(
    FLAGS.gin_config, FLAGS.gin_binding, finalize_config=True
  )

  logging.info(
    f"Writing checkpoints to {FLAGS.checkpoint_dir} "
    f"and summaries to {FLAGS.summary_dir}."
  )

  try:
    train_data, valid_data, test_data = train.load_datasets()
    params, state = train.train(
      train_dataset=train_data,
      valid_dataset=valid_data,
      test_dataset=test_data,
    )
    train.evaluate(
      params=params,
      state=state,
      test_dataset=test_data,
    )

  except ValueError as e:
    logging.info("Full Gin configurations:\n%s", gin.config_str())
    raise e


if __name__ == "__main__":
  logging.get_absl_handler().python_handler.stream = sys.stdout
  app.run(main)
