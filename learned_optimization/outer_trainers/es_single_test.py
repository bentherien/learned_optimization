# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ES-Single loss aggregation modes (_aggregate_delta_losses)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from learned_optimization.outer_trainers.es_single import _aggregate_delta_losses


class AggregateDeltaLossesTest(parameterized.TestCase):
  """Tests for the _aggregate_delta_losses helper function."""

  def setUp(self):
    super().setUp()
    # Simple test data: 5 steps, 3 tasks
    self.delta_losses = jnp.array([
        [0.1, 0.2, 0.3],  # step 0
        [0.2, 0.1, 0.4],  # step 1
        [0.3, 0.3, 0.1],  # step 2
        [0.4, 0.2, 0.2],  # step 3
        [0.5, 0.4, 0.5],  # step 4
    ])  # [5, 3]
    # All steps valid for all tasks
    self.mask = jnp.ones((5, 3))
    self.prev_delta_loss = jnp.zeros(3)

  def test_mean_matches_manual(self):
    """'mean' should return mean of per-step delta losses."""
    delta_loss, new_prev = _aggregate_delta_losses(
        self.delta_losses, self.mask, "mean", self.prev_delta_loss)
    expected = jnp.mean(self.delta_losses, axis=0)
    np.testing.assert_allclose(delta_loss, expected, atol=1e-6)

  def test_sum_matches_manual(self):
    """'sum' should return sum of per-step delta losses."""
    delta_loss, new_prev = _aggregate_delta_losses(
        self.delta_losses, self.mask, "sum", self.prev_delta_loss)
    expected = jnp.sum(self.delta_losses, axis=0)
    np.testing.assert_allclose(delta_loss, expected, atol=1e-6)

  def test_sum_equals_mean_times_steps(self):
    """For uniform mask, sum should equal mean * num_steps."""
    mean_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "mean", self.prev_delta_loss)
    sum_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "sum", self.prev_delta_loss)
    np.testing.assert_allclose(sum_dl, mean_dl * 5.0, atol=1e-6)

  def test_final_uses_last_step(self):
    """'final' should return only the last step's delta loss."""
    delta_loss, new_prev = _aggregate_delta_losses(
        self.delta_losses, self.mask, "final", self.prev_delta_loss)
    expected = self.delta_losses[-1]  # [0.5, 0.4, 0.5]
    np.testing.assert_allclose(delta_loss, expected, atol=1e-6)

  def test_final_only_uses_last_step(self):
    """Verify 'final' ignores earlier steps by making them very different."""
    # Make all steps except last have huge values
    delta_losses = jnp.array([
        [100., 100., 100.],
        [100., 100., 100.],
        [100., 100., 100.],
        [100., 100., 100.],
        [0.1, 0.2, 0.3],  # only this should matter
    ])
    delta_loss, _ = _aggregate_delta_losses(
        delta_losses, self.mask, "final", self.prev_delta_loss)
    np.testing.assert_allclose(delta_loss, jnp.array([0.1, 0.2, 0.3]), atol=1e-6)

  def test_telescoping_first_window_equals_final(self):
    """With prev_delta_loss=0, telescoping should equal final step delta."""
    tele_dl, new_prev = _aggregate_delta_losses(
        self.delta_losses, self.mask, "telescoping", jnp.zeros(3))
    final_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "final", jnp.zeros(3))
    np.testing.assert_allclose(tele_dl, final_dl, atol=1e-6)

  def test_telescoping_cross_window(self):
    """Two sequential telescoping windows should sum to the second window's final."""
    # Window 1
    window1_losses = jnp.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])
    mask1 = jnp.ones((2, 2))
    prev = jnp.zeros(2)

    dl1, new_prev1 = _aggregate_delta_losses(
        window1_losses, mask1, "telescoping", prev)
    # dl1 should be final delta of window 1 = [0.3, 0.4]
    np.testing.assert_allclose(dl1, jnp.array([0.3, 0.4]), atol=1e-6)
    # new_prev1 should be the cumulative final delta = [0.3, 0.4]
    np.testing.assert_allclose(new_prev1, jnp.array([0.3, 0.4]), atol=1e-6)

    # Window 2
    window2_losses = jnp.array([
        [0.5, 0.6],
        [0.7, 0.8],
    ])
    mask2 = jnp.ones((2, 2))

    dl2, new_prev2 = _aggregate_delta_losses(
        window2_losses, mask2, "telescoping", new_prev1)
    # dl2 should be [0.7 - 0.3, 0.8 - 0.4] = [0.4, 0.4]
    np.testing.assert_allclose(dl2, jnp.array([0.4, 0.4]), atol=1e-6)
    # new_prev2 should be [0.7, 0.8]
    np.testing.assert_allclose(new_prev2, jnp.array([0.7, 0.8]), atol=1e-6)

    # Sum of telescoping contributions should equal window 2's final delta
    np.testing.assert_allclose(dl1 + dl2, jnp.array([0.7, 0.8]), atol=1e-6)

  def test_weighted_w0_equals_mean(self):
    """'weighted' with weight=0.0 should equal 'mean'."""
    mean_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "mean", self.prev_delta_loss)
    weighted_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "weighted", self.prev_delta_loss,
        final_loss_weight=0.0)
    np.testing.assert_allclose(weighted_dl, mean_dl, atol=1e-6)

  def test_weighted_w1_equals_final(self):
    """'weighted' with weight=1.0 should equal 'final'."""
    final_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "final", self.prev_delta_loss)
    weighted_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "weighted", self.prev_delta_loss,
        final_loss_weight=1.0)
    np.testing.assert_allclose(weighted_dl, final_dl, atol=1e-6)

  def test_weighted_interpolation(self):
    """'weighted' with weight=0.5 should be average of mean and final."""
    mean_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "mean", self.prev_delta_loss)
    final_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "final", self.prev_delta_loss)
    weighted_dl, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "weighted", self.prev_delta_loss,
        final_loss_weight=0.5)
    expected = 0.5 * mean_dl + 0.5 * final_dl
    np.testing.assert_allclose(weighted_dl, expected, atol=1e-6)


class RaggedMaskTest(parameterized.TestCase):
  """Tests with ragged masks (different valid lengths per task)."""

  def setUp(self):
    super().setUp()
    # 4 steps, 3 tasks. Task 0 has 4 valid, task 1 has 2, task 2 has 3.
    self.delta_losses = jnp.array([
        [1.0, 2.0, 3.0],  # step 0 - valid for all
        [4.0, 5.0, 6.0],  # step 1 - valid for all
        [7.0, 0.0, 9.0],  # step 2 - invalid for task 1
        [10., 0.0, 0.0],  # step 3 - only valid for task 0
    ])
    self.mask = jnp.array([
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 0., 1.],
        [1., 0., 0.],
    ])
    self.prev_delta_loss = jnp.zeros(3)

  def test_mean_with_ragged_mask(self):
    """Mean should only average over valid steps per task."""
    delta_loss, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "mean", self.prev_delta_loss)
    # Task 0: (1+4+7+10)/4 = 5.5
    # Task 1: (2+5)/2 = 3.5
    # Task 2: (3+6+9)/3 = 6.0
    np.testing.assert_allclose(delta_loss, jnp.array([5.5, 3.5, 6.0]), atol=1e-6)

  def test_final_with_ragged_mask(self):
    """Final should pick the last valid step per task."""
    delta_loss, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "final", self.prev_delta_loss)
    # Task 0: step 3 -> 10.0
    # Task 1: step 1 -> 5.0
    # Task 2: step 2 -> 9.0
    np.testing.assert_allclose(delta_loss, jnp.array([10., 5., 9.]), atol=1e-6)

  def test_sum_with_ragged_mask(self):
    """Sum should only include valid steps."""
    delta_loss, _ = _aggregate_delta_losses(
        self.delta_losses, self.mask, "sum", self.prev_delta_loss)
    # Task 0: 1+4+7+10 = 22
    # Task 1: 2+5 = 7
    # Task 2: 3+6+9 = 18
    np.testing.assert_allclose(delta_loss, jnp.array([22., 7., 18.]), atol=1e-6)

  def test_telescoping_with_ragged_mask(self):
    """Telescoping should use the last valid step per task."""
    delta_loss, new_prev = _aggregate_delta_losses(
        self.delta_losses, self.mask, "telescoping", self.prev_delta_loss)
    # Same as final with prev=0: task 0 -> 10, task 1 -> 5, task 2 -> 9
    np.testing.assert_allclose(delta_loss, jnp.array([10., 5., 9.]), atol=1e-6)


class AntitheticInvarianceTest(parameterized.TestCase):
  """Verify antithetic property: swapping pos/neg negates delta_loss."""

  @parameterized.parameters("mean", "sum", "final", "telescoping", "weighted")
  def test_negation_on_swap(self, loss_type):
    """Swapping pos and neg trajectories should negate delta_loss."""
    delta_losses = jnp.array([
        [0.1, -0.2, 0.3],
        [0.4, -0.1, 0.2],
        [0.3, 0.5, -0.1],
    ])
    mask = jnp.ones((3, 3))
    prev = jnp.zeros(3)

    kwargs = dict(final_loss_weight=0.5) if loss_type == "weighted" else {}

    dl_pos, _ = _aggregate_delta_losses(
        delta_losses, mask, loss_type, prev, **kwargs)
    dl_neg, _ = _aggregate_delta_losses(
        -delta_losses, mask, loss_type, -prev, **kwargs)

    np.testing.assert_allclose(dl_pos, -dl_neg, atol=1e-6)


class PrevDeltaLossUnchangedTest(parameterized.TestCase):
  """Verify prev_delta_loss passes through unchanged for non-telescoping modes."""

  @parameterized.parameters("mean", "sum", "final", "weighted")
  def test_prev_unchanged(self, loss_type):
    """Non-telescoping modes should return prev_delta_loss unchanged."""
    delta_losses = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    mask = jnp.ones((2, 2))
    prev = jnp.array([7.0, 8.0])  # arbitrary non-zero values

    kwargs = dict(final_loss_weight=0.5) if loss_type == "weighted" else {}

    _, new_prev = _aggregate_delta_losses(
        delta_losses, mask, loss_type, prev, **kwargs)
    np.testing.assert_allclose(new_prev, prev, atol=1e-6)

  def test_telescoping_updates_prev(self):
    """Telescoping mode should update prev_delta_loss to current final delta."""
    delta_losses = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    mask = jnp.ones((2, 2))
    prev = jnp.array([7.0, 8.0])

    _, new_prev = _aggregate_delta_losses(
        delta_losses, mask, "telescoping", prev)
    # new_prev should be the final step's delta: [0.3, 0.4]
    np.testing.assert_allclose(new_prev, jnp.array([0.3, 0.4]), atol=1e-6)


class WeightedRaggedMaskTest(absltest.TestCase):
  """Test 'weighted' mode with ragged masks."""

  def test_weighted_with_ragged_mask(self):
    """Weighted should correctly blend mean and final with ragged masks."""
    delta_losses = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 0.0, 9.0],
        [10., 0.0, 0.0],
    ])
    mask = jnp.array([
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 0., 1.],
        [1., 0., 0.],
    ])
    prev = jnp.zeros(3)
    # mean: [5.5, 3.5, 6.0], final: [10, 5, 9]
    weighted_dl, _ = _aggregate_delta_losses(
        delta_losses, mask, "weighted", prev, final_loss_weight=0.5)
    expected = 0.5 * jnp.array([5.5, 3.5, 6.0]) + 0.5 * jnp.array([10., 5., 9.])
    np.testing.assert_allclose(weighted_dl, expected, atol=1e-6)


class SingleStepWindowTest(parameterized.TestCase):
  """Tests with single-step windows (shape [1, num_tasks])."""

  @parameterized.parameters("mean", "sum", "final", "telescoping", "weighted")
  def test_single_step(self, loss_type):
    """All modes should work with a single step."""
    delta_losses = jnp.array([[0.5, -0.3]])  # [1, 2]
    mask = jnp.ones((1, 2))
    prev = jnp.zeros(2)
    kwargs = dict(final_loss_weight=0.5) if loss_type == "weighted" else {}
    delta_loss, _ = _aggregate_delta_losses(
        delta_losses, mask, loss_type, prev, **kwargs)
    # All modes should return the single step's value (except sum = same here)
    if loss_type == "sum":
      np.testing.assert_allclose(delta_loss, jnp.array([0.5, -0.3]), atol=1e-6)
    else:
      np.testing.assert_allclose(delta_loss, jnp.array([0.5, -0.3]), atol=1e-6)


class AllZeroMaskTest(parameterized.TestCase):
  """Tests with all-zero mask columns (no valid steps for some tasks)."""

  @parameterized.parameters("mean", "sum", "final", "telescoping", "weighted")
  def test_all_zero_mask_returns_zero(self, loss_type):
    """Tasks with no valid steps should produce zero delta_loss."""
    delta_losses = jnp.array([
        [0.5, 99.0],  # task 1 has garbage values behind mask
        [0.3, 99.0],
    ])
    mask = jnp.array([
        [1., 0.],  # task 0 valid, task 1 fully masked
        [1., 0.],
    ])
    prev = jnp.zeros(2)
    kwargs = dict(final_loss_weight=0.5) if loss_type == "weighted" else {}
    delta_loss, _ = _aggregate_delta_losses(
        delta_losses, mask, loss_type, prev, **kwargs)
    # Task 1 should be 0 (no valid steps)
    self.assertAlmostEqual(float(delta_loss[1]), 0.0, places=5)
    # Task 0 should be finite
    self.assertTrue(jnp.isfinite(delta_loss[0]))


class InvalidLossTypeTest(absltest.TestCase):
  """Verify invalid loss_type raises ValueError."""

  def test_invalid_loss_type(self):
    delta_losses = jnp.array([[0.1], [0.2]])
    mask = jnp.ones((2, 1))
    prev = jnp.zeros(1)
    with self.assertRaises(ValueError):
      _aggregate_delta_losses(delta_losses, mask, "invalid", prev)


if __name__ == "__main__":
  absltest.main()
