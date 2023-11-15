import numpy as np
import unittest

from rules_vs_exemplars.data import base


class RelativeBalancingTest(unittest.TestCase):
  def setUp(self):
    self.condition_mask = np.hstack(
      (
        np.zeros(8 // 2, dtype=bool),
        np.ones(8 // 2, dtype=bool),
      )
    )
    self.discriminator_features = np.hstack(
      (
        np.zeros(8 // 4, dtype=bool),
        np.ones(8 // 4, dtype=bool),
        np.zeros(8 // 4, dtype=bool),
        np.ones(8 // 4, dtype=bool),
      )
    )
    self.distractor_features = np.hstack(
      (
        np.zeros(8 // 8, dtype=bool),
        np.ones(8 // 8, dtype=bool),
        np.zeros(8 // 8, dtype=bool),
        np.ones(8 // 8, dtype=bool),
        np.zeros(8 // 8, dtype=bool),
        np.ones(8 // 8, dtype=bool),
        np.zeros(8 // 8, dtype=bool),
        np.ones(8 // 8, dtype=bool),
      )
    )

  def test_case_masking(self):
    case_masks = base.case_masking(
      self.condition_mask,
      self.discriminator_features,
      self.distractor_features,
    )

    np.testing.assert_array_equal(
      case_masks,
      np.array(
        [
          [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
          ],
          [
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
          ],
        ]
      ),
    )

  def test_compute_case_counts(self):
    case_masks = base.case_masking(
      self.condition_mask,
      self.discriminator_features,
      self.distractor_features,
    )
    case_counts = base.compute_case_counts(case_masks)

    np.testing.assert_array_equal(case_counts, np.array([[1, 1], [1, 1]]))


if __name__ == "__main__":
  unittest.main()
