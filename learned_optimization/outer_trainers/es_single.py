# coding=utf-8
# Copyright 2021 Google LLC
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

"""ES-Single: accumulator-free, unbiased Evolution Strategies gradient estimator.

Implements the algorithm from:
  "Low-Variance Gradient Estimation in Unrolled Computation Graphs with
   ES-Single" (arXiv:2304.11153).

Unlike TruncatedPES, ES-Single does NOT maintain a persistent accumulator of
past gradient contributions. Instead, it fixes a single perturbation vec_pos
at the beginning of each inner problem and reuses it across ALL truncation
windows of that problem:

    grad = mean_over_tasks( vec_pos * delta_loss / (2 * std^2) )

where delta_loss = pos_loss - neg_loss averaged over the current window.

The key invariant (from the reference in snippets/es-single.py): the
perturbation is only resampled when the inner problem resets (is_done). Using
the same noise throughout one inner problem makes the estimator unbiased
(unlike vanilla truncated ES which re-samples each window). No accumulator
state is needed because the perturbation is stored directly in the worker state.
"""

import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

import flax
import gin
import haiku as hk
import jax
from jax import lax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from learned_optimization import jax_utils
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any


@flax.struct.dataclass
class ESSingleWorkerState(gradient_learner.GradientEstimatorState):
  """State for ESSingle.

  Stores pos/neg inner states and the persistent perturbation vec_pos.
  vec_pos is fixed for the entire inner problem and only resampled on reset,
  matching the reference algorithm in snippets/es-single.py.
  """
  pos_state: TruncatedUnrollState
  neg_state: TruncatedUnrollState
  vec_pos: MetaParams  # Persistent perturbation — fixed per inner problem.
  prev_delta_loss: jnp.ndarray  # [num_tasks] cross-window state for telescoping.


# ---------------------------------------------------------------------------
# Loss aggregation for different loss_type modes
# ---------------------------------------------------------------------------

def _aggregate_delta_losses(
    delta_losses: jnp.ndarray,
    mask: jnp.ndarray,
    loss_type: str,
    prev_delta_loss: jnp.ndarray,
    final_loss_weight: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Aggregate per-step delta losses into a per-task scalar for the ES gradient.

  Supports multiple loss aggregation modes:
    - "mean": Mean of per-step delta losses (original ES-Single behavior).
    - "sum": Sum of per-step delta losses (no normalization).
    - "final": Only the last valid timestep's delta loss.
    - "telescoping": Cross-window telescoping sum targeting final loss. Each
      window's contribution is the change in cumulative delta loss from the
      previous window's end, so that across all windows of an inner problem
      the gradient updates telescope to the true final-loss gradient.
    - "weighted": Weighted blend of mean and final losses, controlled by
      final_loss_weight (0 = pure mean, 1 = pure final).

  Args:
    delta_losses: [steps, num_tasks] per-step pos-neg loss differences.
    mask: [steps, num_tasks] validity mask (1 for valid steps).
    loss_type: Aggregation mode (static, for JIT branching).
    prev_delta_loss: [num_tasks] previous window's cumulative final delta
      (used only in "telescoping" mode).
    final_loss_weight: Blending weight for "weighted" mode.

  Returns:
    (delta_loss, new_prev_delta_loss): Per-task scalar delta loss and updated
      prev_delta_loss (only changes in "telescoping" mode).
  """
  num_tasks = delta_losses.shape[1]
  denom = jnp.sum(mask, axis=0)  # [num_tasks]
  task_indices = jnp.arange(num_tasks)

  # Guard: clamp denom to >= 1 to avoid division by zero with all-zero masks.
  safe_denom = jnp.maximum(denom, 1.0)

  if loss_type == "mean":
    delta_loss = jnp.sum(delta_losses * mask, axis=0) / safe_denom
    return delta_loss, prev_delta_loss

  elif loss_type == "sum":
    delta_loss = jnp.sum(delta_losses * mask, axis=0)
    return delta_loss, prev_delta_loss

  elif loss_type == "final":
    last_idx = jnp.maximum(denom.astype(jnp.int32) - 1, 0)  # [num_tasks]
    delta_loss = delta_losses[last_idx, task_indices]
    # Zero out tasks with no valid steps (all-zero mask).
    delta_loss = jnp.where(denom > 0, delta_loss, 0.0)
    return delta_loss, prev_delta_loss

  elif loss_type == "telescoping":
    # Cumulative delta at the end of this window.
    last_idx = jnp.maximum(denom.astype(jnp.int32) - 1, 0)
    current_final_delta = delta_losses[last_idx, task_indices]
    # Zero out tasks with no valid steps.
    current_final_delta = jnp.where(denom > 0, current_final_delta, 0.0)
    # Window contribution = change from previous window's end.
    delta_loss = current_final_delta - prev_delta_loss
    return delta_loss, current_final_delta

  elif loss_type == "weighted":
    mean_dl = jnp.sum(delta_losses * mask, axis=0) / safe_denom
    last_idx = jnp.maximum(denom.astype(jnp.int32) - 1, 0)
    final_dl = delta_losses[last_idx, task_indices]
    final_dl = jnp.where(denom > 0, final_dl, 0.0)
    delta_loss = (1.0 - final_loss_weight) * mean_dl + final_loss_weight * final_dl
    return delta_loss, prev_delta_loss

  else:
    raise ValueError(f"Unsupported loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# Single-machine gradient computation
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("std", "timer_obj", "sign_delta_loss_scalar",
                     "samples_per_device", "device_idx", "loss_type"))
def compute_es_single_grad(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: Optional[int] = None,  # unused; kept for API parity
    device_idx: Optional[int] = None,           # unused; kept for API parity
    loss_type: str = "mean",
    prev_delta_loss: Optional[jnp.ndarray] = None,
    final_loss_weight: float = 0.0,
) -> Tuple[float, MetaParams, truncated_step_mod.TruncatedUnrollOut,
           jnp.ndarray, jnp.ndarray]:
  """Compute ES-Single gradient estimate on a single machine.

  Gradient formula (per task t):
      delta_loss_t = aggregate( (p_loss - n_loss) * mask )  [mode: loss_type]
      es_grad      = mean_over_tasks( vec_pos_t * delta_loss_t / (2 * std^2) )

  Args:
    p_yses: Sequence of TruncatedUnrollOut from the positive perturbation.
    n_yses: Sequence of TruncatedUnrollOut from the negative perturbation.
    vec_pos: Positive perturbations, shape [num_tasks, *theta_shape].
    std: Standard deviation of perturbations.
    timer_obj: Timer context (static, for profiling; no-op body here).
    sign_delta_loss_scalar: If set, replace delta_loss with its sign * scalar.
    samples_per_device: Unused in single-machine mode; present for API parity.
    device_idx: Unused in single-machine mode; present for API parity.
    loss_type: Aggregation mode ("mean", "sum", "final", "telescoping",
      "weighted"). Static for JIT.
    prev_delta_loss: [num_tasks] previous window's cumulative delta (telescoping).
    final_loss_weight: Blending weight for "weighted" mode.

  Returns:
    (loss, es_grad, p_ys, delta_losses, new_prev_delta_loss)
  """
  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  with timer_obj("ES-Single Gather", []):
    pass  # No cross-device communication in single-machine mode.

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # delta_losses: [steps, num_tasks]
  delta_losses = p_ys.loss - n_ys.loss

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = (
        jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar)

  # Aggregate delta losses according to loss_type.
  delta_loss, new_prev_delta_loss = _aggregate_delta_losses(
      delta_losses, p_ys.mask, loss_type, prev_delta_loss, final_loss_weight)

  factor = 1.0 / (2 * std**2)
  num_tasks = delta_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  vec_es_grad = jax.tree_util.tree_map(
      functools.partial(reshape_to, delta_loss), vec_pos)
  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  # Reporting loss: always use mean for consistent logging across loss_types.
  denom = jnp.sum(p_ys.mask, axis=0)
  pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / denom
  neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / denom
  loss = jnp.mean((pos_loss + neg_loss) / 2.0)

  return loss, es_grad, p_ys, delta_losses, new_prev_delta_loss


# ---------------------------------------------------------------------------
# Multi-GPU (sharded) gradient computation — mirrors compute_pes_grad_sharded
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("std", "sign_delta_loss_scalar", "replicated",
                     "loss_type"))
def _es_single_grad_sharded_inner(p_ys, n_ys, vec_pos, std,
                                   sign_delta_loss_scalar, replicated,
                                   loss_type="mean",
                                   prev_delta_loss=None,
                                   final_loss_weight=0.0):
  """JIT-compiled all-gather via sharding constraint, then ES-Single gradient."""
  # All-gather: replicate all sharded arrays across devices via constraint.
  p_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), p_ys)
  n_ys = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), n_ys)
  vec_pos = jax.tree_util.tree_map(
      lambda x: lax.with_sharding_constraint(x, replicated), vec_pos)
  prev_delta_loss = lax.with_sharding_constraint(prev_delta_loss, replicated)

  delta_losses = p_ys.loss - n_ys.loss

  if sign_delta_loss_scalar:
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = (
        jnp.ones_like(delta_losses) * sign_per_task * sign_delta_loss_scalar)

  # Aggregate delta losses according to loss_type.
  delta_loss, new_prev_delta_loss = _aggregate_delta_losses(
      delta_losses, p_ys.mask, loss_type, prev_delta_loss, final_loss_weight)

  factor = 1.0 / (2 * std**2)
  num_tasks = delta_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  vec_es_grad = jax.tree_util.tree_map(
      functools.partial(reshape_to, delta_loss), vec_pos)
  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  # Reporting loss: always use mean for consistent logging.
  denom = jnp.sum(p_ys.mask, axis=0)
  pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / denom
  neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / denom
  loss = jnp.mean((pos_loss + neg_loss) / 2.0)

  return loss, es_grad, p_ys, delta_losses, new_prev_delta_loss


def compute_es_single_grad_sharded(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: int = 1,
    device_idx: int = 0,
    mesh: Optional[Mesh] = None,
    loss_type: str = "mean",
    prev_delta_loss: Optional[jnp.ndarray] = None,
    final_loss_weight: float = 0.0,
) -> Tuple[float, MetaParams, truncated_step_mod.TruncatedUnrollOut,
           jnp.ndarray, jnp.ndarray]:
  """Compute ES-Single gradient using JAX mesh-based sharding for multi-GPU.

  Follows the same all-gather pattern as compute_pes_grad_sharded in
  truncated_pes.py. Each process holds a local shard of particles; a globally-
  sharded array is constructed so that jit + with_sharding_constraint triggers
  NCCL all-gather, then per-device results are sliced back out.

  Args:
    p_yses: Local TruncatedUnrollOut sequence from positive perturbation.
    n_yses: Local TruncatedUnrollOut sequence from negative perturbation.
    vec_pos: Local positive perturbations, shape [local_tasks, *theta_shape].
    std: Standard deviation of perturbations.
    timer_obj: Timer context (static) for profiling.
    sign_delta_loss_scalar: Optional sign-based delta loss scaling.
    samples_per_device: Number of tasks owned by this process.
    device_idx: Index of the current process (jax.process_index()).
    mesh: JAX Mesh created in ESSingle.__init__.
    loss_type: Aggregation mode (see _aggregate_delta_losses).
    prev_delta_loss: [local_tasks] previous window's cumulative delta.
    final_loss_weight: Blending weight for "weighted" mode.

  Returns:
    (loss, es_grad, p_ys, delta_losses, new_prev_delta_loss)
  """
  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  replicated = NamedSharding(mesh, P())
  sharding_axis1 = NamedSharding(mesh, P(None, 'devices'))
  sharding_axis0 = NamedSharding(mesh, P('devices'))

  num_devices = jax.device_count()
  local_device = jax.local_devices()[0]

  def _to_single_device(arr):
    """Extract the local shard if the array spans multiple devices."""
    if isinstance(arr, jax.Array) and not arr.is_fully_addressable:
      return arr.addressable_data(0)
    return jax.device_put(arr, local_device)

  def make_global_axis1(local_arr):
    """[steps, local_tasks] -> global [steps, total_tasks] sharded on axis 1."""
    global_shape = list(local_arr.shape)
    global_shape[1] = global_shape[1] * num_devices
    local_arr = _to_single_device(local_arr)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape), sharding_axis1, [local_arr])

  def make_global_axis0(local_arr):
    """[local_tasks, ...] -> global [total_tasks, ...] sharded on axis 0."""
    global_shape = list(local_arr.shape)
    global_shape[0] = global_shape[0] * num_devices
    local_arr = _to_single_device(local_arr)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape), sharding_axis0, [local_arr])

  with timer_obj("ES-Single Gather", []):
    p_ys = jax.tree_util.tree_map(make_global_axis1, p_ys)
    n_ys = jax.tree_util.tree_map(make_global_axis1, n_ys)
    vec_pos = jax.tree_util.tree_map(make_global_axis0, vec_pos)
    prev_delta_loss = make_global_axis0(prev_delta_loss)

  loss, es_grad, p_ys_out, delta_losses, new_prev_delta_loss = (
      _es_single_grad_sharded_inner(
          p_ys, n_ys, vec_pos,
          std=std,
          sign_delta_loss_scalar=sign_delta_loss_scalar,
          replicated=replicated,
          loss_type=loss_type,
          prev_delta_loss=prev_delta_loss,
          final_loss_weight=final_loss_weight,
      ))

  # Slice back the per-device portion of per-task outputs.
  start_idx = device_idx * samples_per_device
  end_idx = start_idx + samples_per_device

  p_ys_out = jax.tree_util.tree_map(
      lambda x: x[:, start_idx:end_idx] if hasattr(x, 'shape') else x,
      p_ys_out)
  delta_losses = delta_losses[:, start_idx:end_idx]
  new_prev_delta_loss = new_prev_delta_loss[start_idx:end_idx]

  return loss, es_grad, p_ys_out, delta_losses, new_prev_delta_loss


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@gin.configurable
class ESSingle(gradient_learner.GradientEstimator):
  """GradientEstimator using ES-Single (accumulator-free truncated ES).

  ES-Single treats every truncation window independently.  For each window the
  gradient estimate is:

      grad = mean_over_tasks( vec_pos * delta_loss / (2 * std^2) )

  where delta_loss = mask-weighted mean of (pos_loss - neg_loss) over the
  window steps.

  Compared to TruncatedPES:
  - **No accumulator** — state is (pos_state, neg_state, vec_pos), not
    (pos_state, neg_state, accumulator).
  - **Unbiased** — the perturbation is fixed per inner problem, so truncation
    bias is eliminated (see arXiv:2304.11153).
  - **Lower variance** — no amplification from accumulated perturbations.
  - **Simpler** — no accumulator to track; vec_pos is reset only on is_done.

  Constructor arguments intentionally mirror TruncatedPES so the two
  estimators are drop-in replaceable in meta_trainers.py.
  """

  _VALID_LOSS_TYPES = ("mean", "sum", "final", "telescoping", "weighted")

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      trunc_length: int = 10,
      std: float = 0.01,
      steps_per_jit: int = 10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
      trunc_schedule=None,
      pmap_across_devices: bool = False,
      timer_obj: Any = None,
      use_bc_grads: bool = False,
      std_schedule=None,
      loss_type: str = "mean",
      final_loss_weight: float = 0.0,
  ):
    self.truncated_step = truncated_step
    self.std = std
    self.trunc_length = trunc_length
    self.steps_per_jit = steps_per_jit
    self.stack_antithetic_samples = stack_antithetic_samples
    self.sign_delta_loss_scalar = sign_delta_loss_scalar
    self.trunc_schedule = trunc_schedule
    self.update_truncation_length(0)
    self.samples_per_device = self.truncated_step.num_tasks
    self.use_bc_grads = use_bc_grads
    self.std_schedule = std_schedule
    self.loss_type = loss_type
    self.final_loss_weight = final_loss_weight

    if loss_type not in self._VALID_LOSS_TYPES:
      raise ValueError(
          f"loss_type must be one of {self._VALID_LOSS_TYPES}, "
          f"got {loss_type!r}")
    if loss_type == "weighted" and not (0.0 <= final_loss_weight <= 1.0):
      raise ValueError(
          f"final_loss_weight must be in [0, 1] for 'weighted' loss_type, "
          f"got {final_loss_weight}")

    if pmap_across_devices:
      devices = mesh_utils.create_device_mesh((jax.device_count(),))
      self.mesh = Mesh(devices, axis_names=('devices',))
      self.grad_fn = functools.partial(
          compute_es_single_grad_sharded, mesh=self.mesh)
    else:
      self.mesh = None
      self.grad_fn = compute_es_single_grad

    self.timer_obj = timer_obj
    assert self.timer_obj is not None, "timer_obj must be provided"

    if self.trunc_length % self.steps_per_jit != 0:
      raise ValueError(
          "trunc_length must be an integer multiple of steps_per_jit. "
          f"Got trunc_length={trunc_length}, steps_per_jit={steps_per_jit}.")

  def update_std(self, iteration):
    """Update std from the optional schedule."""
    if self.std_schedule is not None:
      self.std = self.std_schedule(iteration)
      # Also update the grad function if using sharded mode

  def task_name(self) -> str:
    return self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> ESSingleWorkerState:
    """Initialise inner-problem states and sample the initial perturbation."""
    theta = worker_weights.theta
    rng = hk.PRNGSequence(key)
    pos_unroll_state = self.truncated_step.init_step_state(
        theta, worker_weights.outer_state, next(rng), theta_is_vector=False)
    neg_unroll_state = pos_unroll_state
    vec_pos, _, _ = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)
    return ESSingleWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        vec_pos=vec_pos,
        prev_delta_loss=jnp.zeros(
            self.truncated_step.num_tasks, dtype=jnp.float32))

  def update_truncation_length(self, iteration):
    """Update trunc_length from the optional schedule."""
    if self.trunc_schedule is not None:
      self.trunc_length = self.trunc_schedule(iteration)

  @profile.wrap()
  def get_datas(self):
    return [
        self.truncated_step.get_batch(self.steps_per_jit)
        for _ in range(self.trunc_length // self.steps_per_jit)
    ]

  @profile.wrap()
  def compute_gradient_estimate(  # pytype: disable=signature-mismatch
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: ESSingleWorkerState,
      with_summary: bool = False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    p_state = state.pos_state
    n_state = state.neg_state
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    # ES-Single: reuse the perturbation fixed at the start of the inner problem.
    # vec_pos is persistent in state and only resampled when a task resets.
    #
    # Device note: in multi-GPU sharded mode, the sharded grad path uses
    # lax.with_sharding_constraint(..., replicated) which places p_ys on all
    # global GPUs.  To prevent that global sharding from propagating into
    # vec_pos (and thus into vec_p_theta on the NEXT iteration), we pin vec_pos
    # to the local device here.  This is safe because each process owns a
    # disjoint subset of tasks and vec_pos values are the same on all replicas.
    #
    # We use addressable_data(0) for globally-sharded arrays (device_put only
    # works on fully-addressable arrays), falling back to device_put otherwise.
    _local_dev = jax.local_devices()[0]

    def _to_local(x):
      if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return x.addressable_data(0)
      return jax.device_put(x, _local_dev)

    vec_pos = jax.tree_util.tree_map(_to_local, state.vec_pos)
    prev_delta_loss = _to_local(state.prev_delta_loss)
    vec_p_theta = jax.tree_util.tree_map(lambda t, p: t + p, theta, vec_pos)
    vec_n_theta = jax.tree_util.tree_map(lambda t, p: t - p, theta, vec_pos)

    p_yses = []
    n_yses = []
    p_bc_grads_list = []
    n_bc_grads_list = []
    metrics = []

    for i in range(self.trunc_length // self.steps_per_jit):
      if datas_list is None:
        if jax_utils.in_jit():
          raise ValueError("Must pass data in when using a jit gradient est.")
        datas = self.truncated_step.get_batch(self.steps_per_jit)
      else:
        datas = datas_list[i]

      # Force strong types (avoids JIT retracing due to weak-type scalars).
      p_state = jax.tree_util.tree_map(
          lambda x: jnp.asarray(x, dtype=x.dtype), p_state)
      n_state = jax.tree_util.tree_map(
          lambda x: jnp.asarray(x, dtype=x.dtype), n_state)

      key = next(rng)
      p_state, n_state, p_ys, n_ys, p_bc_grads, n_bc_grads, m = (
          common.maybe_stacked_es_unroll(
              self.truncated_step,
              self.steps_per_jit,
              self.stack_antithetic_samples,
              vec_p_theta,
              vec_n_theta,
              p_state,
              n_state,
              key,
              datas,
              worker_weights.outer_state,
              with_summary=with_summary,
              sample_rng_key=next(rng)))

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)
      p_bc_grads_list.append(p_bc_grads)
      n_bc_grads_list.append(n_bc_grads)

    if self.use_bc_grads:
      stacked_bc_grads = jax.tree_util.tree_map(
          lambda *xs: jnp.stack(xs),
          *(p_bc_grads_list + n_bc_grads_list))
      mean_bc_grads = jax.tree_util.tree_map(
          lambda x: jnp.mean(x, axis=(0, 1, 2)), stacked_bc_grads)
    else:
      mean_bc_grads = None

    # ES-Single gradient with configurable loss aggregation.
    loss, es_grad, p_ys, delta_loss, new_prev_delta_loss = self.grad_fn(
        p_yses,
        n_yses,
        vec_pos,
        self.std,
        self.timer_obj,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar,
        samples_per_device=self.samples_per_device,
        device_idx=jax.process_index(),
        loss_type=self.loss_type,
        prev_delta_loss=prev_delta_loss,
        final_loss_weight=self.final_loss_weight)

    # Resample perturbations for tasks whose inner problem reset this window.
    # Tasks where is_done fired get fresh noise; others keep the same vec_pos.
    # This matches the reference (snippets/es-single.py): key only splits on reset.
    #
    # Device note: p_ys.is_done may be globally-sharded (all GPUs) because it
    # went through lax.with_sharding_constraint in the sharded grad path.  Pin
    # it to the local device before computing did_reset so that updated_vec_pos
    # stays local and doesn't infect vec_p_theta on the next iteration.
    is_done_local = _to_local(p_ys.is_done)
    did_reset = jnp.any(is_done_local, axis=0)  # [num_tasks]
    new_vec_pos, _, _ = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)

    def _select(old, new):
      mask = did_reset.reshape((did_reset.shape[0],) + (1,) * (len(old.shape) - 1))
      return jnp.where(mask, new, old)

    updated_vec_pos = jax.tree_util.tree_map(_select, vec_pos, new_vec_pos)

    # Pin new_prev_delta_loss to local device (may be globally-sharded after
    # the sharded grad path) and reset to zero for tasks that reset.
    new_prev_delta_loss = _to_local(new_prev_delta_loss)
    updated_prev_delta_loss = jnp.where(did_reset, 0.0, new_prev_delta_loss)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        bc_grad=mean_bc_grads,
        unroll_state=ESSingleWorkerState(
            p_state, n_state, updated_vec_pos, updated_prev_delta_loss),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(
        metrics, use_jnp=jax_utils.in_jit(), key=next(rng))
    if with_summary:
      # ES-Single has no baseline; log 0.0 to keep the metric key consistent.
      metrics["sample||baseline_loss"] = 0.0
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
