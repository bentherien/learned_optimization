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

"""A vetorized, truncated, PES based gradient estimator."""

import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

import flax
from flax import jax_utils as flax_jax_utils
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
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
class PESWorkerState(gradient_learner.GradientEstimatorState):
  pos_state: TruncatedUnrollState
  neg_state: TruncatedUnrollState
  accumulator: MetaParams


# @functools.partial(jax.jit, static_argnames=("std", "sign_delta_loss_scalar"))
def compute_pes_grad_pmap(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: int = 1,
    device_idx: int = 0,
    baseline_losses: Optional[list[float]] = None,
) -> Tuple[float, MetaParams, MetaParams, truncated_step_mod.TruncatedUnrollOut,
           float]:
  """Compute the PES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of PES outputs from the positive perturbation.
    n_yses: Sequence of PES outputs from the negative perturbation.
    accumulator: Current PES accumulator from the last iteration.
    vec_pos: Positive perturbations used to compute the current unroll.
    std: Standard deviation of pertrubations used.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss

  Returns:
    loss: the mean loss.
    es_grad: the grad estimate.
    new_accumulator: the new accumulator value.
    delta_loss: the difference in positive and negative losses.

  """
  def allgather_pytree(pytree, axis=0):
    """
    Perform an all-gather operation on all leaf tensors in the pytree.
    The tensors are stacked along dimension 0.
    """
    return jax.tree_util.tree_map(lambda x: jax.lax.all_gather(x, 'devices', axis=axis), pytree)

  def reshape_first_three_dims(x):
    # Get shape and reshape to combine first two dims
    shape = x.shape
    return x.reshape([shape[0] * shape[1] * shape[2]] + list(shape[3:]))

  def reshape_last_two_dims(x):
    # Get shape and reshape to combine first two dims
    shape = x.shape
    return x.reshape(list(shape[:1]) + [shape[1] * shape[2]] )

  ####################################################################################
  # start gather
  ####################################################################################
  with timer_obj("PES Gather", []):
    p_yses = jax.pmap(functools.partial(allgather_pytree, axis=1), axis_name='devices')(p_yses)
    n_yses = jax.pmap(functools.partial(allgather_pytree, axis=1), axis_name='devices')(n_yses)

    p_yses = jax.tree_util.tree_map(reshape_last_two_dims, p_yses)
    n_yses = jax.tree_util.tree_map(reshape_last_two_dims, n_yses)

    # if jax.process_index() == 0:
    #   print("p_yses shapes:", jax.tree_util.tree_map(lambda x: x.shape, p_yses))
    #   print("n_yses shapes:", jax.tree_util.tree_map(lambda x: x.shape, n_yses))

    accumulator = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), accumulator)
    vec_pos = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), vec_pos)

    accumulator = jax.pmap(functools.partial(allgather_pytree, axis=0), axis_name='devices')(accumulator)
    vec_pos = jax.pmap(functools.partial(allgather_pytree, axis=0), axis_name='devices')(vec_pos)

    accumulator = jax.tree_util.tree_map(reshape_first_three_dims, accumulator)
    vec_pos = jax.tree_util.tree_map(reshape_first_three_dims, vec_pos)


  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  
  #Flatten tensorse
  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  #get the direction (L(\theta + \eps) - L(\theta - \eps))
  delta_losses = p_ys.loss - n_ys.loss

  if sign_delta_loss_scalar:
    # With PES, there is no single loss for a truncation. For the particular
    # perturbation we will estimate the sign by first averaging.
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = jnp.ones_like(
        delta_losses) * sign_per_task * sign_delta_loss_scalar


  has_finished = lax.cumsum(jnp.asarray(p_ys.is_done, dtype=jnp.int32)) > 0

  # p_ys is of the form [sequence, n_tasks]
  denom = jnp.sum(p_ys.mask, axis=0)

  last_unroll_loss = jnp.sum(
      delta_losses * (1.0 - has_finished) * p_ys.mask, axis=0) / denom

  new_unroll_loss = jnp.sum(
      delta_losses * has_finished * p_ys.mask, axis=0) / denom

  factor = 1.0 / (2 * std**2)

  accumulator = tree_utils.tree_add(vec_pos, accumulator)

  num_tasks = last_unroll_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  es_grad_from_accum = jax.tree_util.tree_map(
      functools.partial(reshape_to, last_unroll_loss), accumulator)

  es_grad_from_new_perturb = jax.tree_util.tree_map(
      functools.partial(reshape_to, new_unroll_loss), vec_pos)

  vec_es_grad = jax.tree_util.tree_map(lambda a, b: a + b, es_grad_from_accum,
                                       es_grad_from_new_perturb)

  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  def _switch_one_accum(a, b):
    shape = [num_tasks] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos,
                                           accumulator)

  pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  # if jax.process_index() == 0:
  #   print("="*100)
  #   print("before output")
  #   print("="*100)
  #   print("pos_loss shape:", pos_loss.shape)
  #   print("neg_loss shape:", neg_loss.shape)
  #   print("es_grad shape:", jax.tree_util.tree_map(lambda x: x.shape, es_grad))
  #   print("new_accumulator shape:", jax.tree_util.tree_map(lambda x: x.shape, new_accumulator))
  #   print("p_ys shape:", jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, p_ys))
  #   print("delta_losses shape:", delta_losses.shape)


  #   pos_loss shape: (2,)
  # neg_loss shape: (2,)
  # es_grad shape: {'adafactor_decays': (3,), 'momentum_decays': (3,), 'nn': {'~': {'b0': (32,), 'b1': (32,), 'b2': (2,), 'w0': (39, 32), 'w1': (32, 32), 'w2': (32, 2)}}, 'rms_decays': (1,)}
  # new_accumulator shape: {'adafactor_decays': (2, 3), 'momentum_decays': (2, 3), 'nn': {'~': {'b0': (2, 32), 'b1': (2, 32), 'b2': (2, 2), 'w0': (2, 39, 32), 'w1': (2, 32, 32), 'w2': (2, 32, 2)}}, 'rms_decays': (2, 1)}
  # p_ys shape: TruncatedUnrollOut(loss=(50, 2), is_done=(50, 2), task_param=(50, 2), iteration=(50, 2), mask=(50, 2))
  # delta_losses shape: (50, 2)
  # Get current device index and total number of devices
  # device_idx = jax.process_index()
  # num_devices = jax.device_count()

  # Calculate samples per device (total samples = 2)
  # samples_per_device = 2 // num_devices

  # Get slice indices for current device
  start_idx = device_idx * samples_per_device
  end_idx = start_idx + samples_per_device

  # Slice outputs to keep only samples for current device
  # pos_loss = pos_loss[start_idx:end_idx]
  # neg_loss = neg_loss[start_idx:end_idx]
  
  # Helper function to slice pytrees
  def slice_first_dim(x):
    if hasattr(x, 'shape'):
      return x[start_idx:end_idx]
    return x

  # Slice pytree outputs
  new_accumulator = jax.tree_util.tree_map(slice_first_dim, new_accumulator)
  p_ys = jax.tree_util.tree_map(lambda x: x[:, start_idx:end_idx] if hasattr(x, 'shape') else x, p_ys)
  delta_losses = delta_losses[:, start_idx:end_idx]


  # print("pos_loss shape:", pos_loss.shape)
  # print("neg_loss shape:", neg_loss.shape)
  # print("es_grad shape:", jax.tree_util.tree_map(lambda x: x.shape, es_grad))
  # print("new_accumulator shape:", jax.tree_util.tree_map(lambda x: x.shape, new_accumulator))
  # print("p_ys shape:", jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, p_ys))
  # print("delta_losses shape:", delta_losses.shape)
  # exit(0)
  # if jax.process_index() == 0:
  #   print("="*100)
  #   print("after output")
  #   print("="*100)
  #   print("start_idx, end_idx:", start_idx, end_idx, "samples_per_device:", samples_per_device, "device_idx:", device_idx)
  #   print("pos_loss shape:", pos_loss.shape)
  #   print("neg_loss shape:", neg_loss.shape)
  #   print("es_grad shape:", jax.tree_util.tree_map(lambda x: x.shape, es_grad))
  #   print("new_accumulator shape:", jax.tree_util.tree_map(lambda x: x.shape, new_accumulator))
  #   print("p_ys shape:", jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, p_ys))
  #   print("delta_losses shape:", delta_losses.shape)


  return (
      jnp.mean((pos_loss + neg_loss) / 2.0),
      0.0,
      es_grad,
      new_accumulator,
      p_ys,
      delta_losses,
  )  # pytype: disable=bad-return-type



@functools.partial(jax.jit, static_argnames=("std", "timer_obj", "sign_delta_loss_scalar", "samples_per_device", "device_idx"))
def compute_pes_grad(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    timer_obj: Any,
    sign_delta_loss_scalar: Optional[float] = None,
    samples_per_device: Optional[int] = None,
    device_idx: Optional[int] = None,
    baseline_losses: Optional[list[float]] = None,
) -> Tuple[float, MetaParams, MetaParams, truncated_step_mod.TruncatedUnrollOut,
           float]:
  """Compute the PES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of PES outputs from the positive perturbation.
    n_yses: Sequence of PES outputs from the negative perturbation.
    accumulator: Current PES accumulator from the last iteration.
    vec_pos: Positive perturbations used to compute the current unroll.
    std: Standard deviation of pertrubations used.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss

  Returns:
    loss: the mean loss.
    es_grad: the grad estimate.
    new_accumulator: the new accumulator value.
    delta_loss: the difference in positive and negative losses.

  """

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))


  
  with timer_obj("PES Gather", []):
    pass

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))


  
  # import pdb; pdb.set_trace()
  

  if baseline_losses is not None:

    # index into baseline_losses using p_ys.iteration
    b = jnp.take(baseline_losses, p_ys.iteration, axis=0)
    delta_losses = (p_ys.loss - b) - (n_ys.loss - b)

    pos_loss_b = jnp.sum((p_ys.loss - b) * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
    neg_loss_b = jnp.sum((n_ys.loss - b) * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)
    b_loss = jnp.mean((pos_loss_b + neg_loss_b) / 2.0)
  else:
    b_loss = 0.0
    delta_losses = p_ys.loss - n_ys.loss



  if sign_delta_loss_scalar:
    # With PES, there is no single loss for a truncation. For the particular
    # perturbation we will estimate the sign by first averaging.
    sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
    delta_losses = jnp.ones_like(
        delta_losses) * sign_per_task * sign_delta_loss_scalar

  has_finished = lax.cumsum(jnp.asarray(p_ys.is_done, dtype=jnp.int32)) > 0

  # p_ys is of the form [sequence, n_tasks]
  denom = jnp.sum(p_ys.mask, axis=0)

  last_unroll_loss = jnp.sum(
      delta_losses * (1.0 - has_finished) * p_ys.mask, axis=0) / denom

  new_unroll_loss = jnp.sum(
      delta_losses * has_finished * p_ys.mask, axis=0) / denom

  factor = 1.0 / (2 * std**2)

  accumulator = tree_utils.tree_add(vec_pos, accumulator)

  num_tasks = last_unroll_loss.shape[0]

  def reshape_to(loss, p):
    return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

  es_grad_from_accum = jax.tree_util.tree_map(
      functools.partial(reshape_to, last_unroll_loss), accumulator)

  es_grad_from_new_perturb = jax.tree_util.tree_map(
      functools.partial(reshape_to, new_unroll_loss), vec_pos)

  vec_es_grad = jax.tree_util.tree_map(lambda a, b: a + b, es_grad_from_accum,
                                       es_grad_from_new_perturb)

  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  def _switch_one_accum(a, b):
    shape = [num_tasks] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos,
                                           accumulator)

  pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

  return (
      jnp.mean((pos_loss + neg_loss) / 2.0),
      b_loss,
      es_grad,
      new_accumulator,
      p_ys,
      delta_losses,
  )  # pytype: disable=bad-return-type

@gin.configurable
class TruncatedPES(gradient_learner.GradientEstimator):
  """GradientEstimator for computing PES gradient estimates.

  Persistent Evolution Strategies (PES) is a gradient estimation technique
  for computing unbiased gradients in a unrolled computation graph. It does this
  by building of of Evolutionary Strategies but additionally keeping a running
  buffer of all the previously used perturbations. See the paper for more
  details (http://proceedings.mlr.press/v139/vicol21a.html).

  In practice, PES is higher variance than pure truncated ES but lower bias.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      trunc_length=10,
      std=0.01,
      steps_per_jit=10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
      trunc_schedule = None,
      pmap_across_devices: bool = False,
      timer_obj: Any = None,
      use_bc_grads: bool = False,
      baseline_losses: Optional[list[float]] = None,
      use_baseline_losses: bool = False,
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
    self.grad_fn = compute_pes_grad_pmap if pmap_across_devices else compute_pes_grad
    self.timer_obj = timer_obj
    self.use_bc_grads = use_bc_grads
    self.use_baseline_losses = use_baseline_losses

    
    self.baseline_losses = baseline_losses
    if self.use_baseline_losses is not None:
      self.baseline_losses = jnp.array(self.baseline_losses, device=jax.devices()[jax.process_index()])

    assert self.timer_obj is not None, "timer_obj must be provided"

    if self.trunc_length % self.steps_per_jit != 0:
      raise ValueError("Pass a trunc_length and steps_per_jit that are"
                       " multiples of each other.")

  def task_name(self) -> str:
    return self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    theta = worker_weights.theta

    pos_unroll_state = self.truncated_step.init_step_state(
        theta, worker_weights.outer_state, key, theta_is_vector=False)
    neg_unroll_state = pos_unroll_state

    accumulator = jax.tree_util.tree_map(
        lambda x: jnp.zeros([self.truncated_step.num_tasks] + list(x.shape)),
        theta)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  def update_truncation_length(self, iteration):
    if self.trunc_schedule is not None:
      self.trunc_length = self.trunc_schedule(iteration)
      # print(f"update_truncation_length() trunc_length={self.trunc_length}")

  @profile.wrap()
  def get_datas(self):
    return [
        self.truncated_step.get_batch(self.steps_per_jit)
        for _ in range(self.trunc_length // self.steps_per_jit)
    ]

  @profile.wrap()
  def compute_gradient_estimate(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)

    p_yses = []
    n_yses = []
    p_bc_grads_list = []
    n_bc_grads_list = []
    metrics = []

    # TODO(lmetz) consider switching this to be a jax.lax.scan when inside jit.
    for i in range(self.trunc_length // self.steps_per_jit):
      if datas_list is None:
        if jax_utils.in_jit():
          raise ValueError("Must pass data in when using a jit gradient est.")
        datas = self.truncated_step.get_batch(self.steps_per_jit)
      else:
        datas = datas_list[i]

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

                  

      key = next(rng)

      p_state, n_state, p_ys, n_ys, p_bc_grads, n_bc_grads, m = common.maybe_stacked_es_unroll(
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
          sample_rng_key=next(rng))

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)
      
      p_bc_grads_list.append(p_bc_grads)
      n_bc_grads_list.append(n_bc_grads)



    if self.use_bc_grads:
      # Convert lists of gradients to stacked arrays
      stacked_bc_grads = jax.tree_util.tree_map(
          lambda *xs: jnp.stack(xs),
          *(p_bc_grads_list + n_bc_grads_list)
      )
      # print(jax.tree_util.tree_map(lambda x: x.shape, stacked_bc_grads))
      
      # Take the mean along the first two dimensions
      mean_bc_grads = jax.tree_util.tree_map(
          lambda x: jnp.mean(x, axis=(0, 1, 2)),
          stacked_bc_grads
      )
      # print(jax.tree_util.tree_map(lambda x: x.shape, mean_bc_grads))
      # exit(0)
    else:
      mean_bc_grads = None


    loss, b_loss, es_grad, new_accumulator, p_ys, delta_loss = self.grad_fn(
        p_yses,
        n_yses,
        accumulator,
        vec_pos,
        self.std,
        self.timer_obj,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar,
        samples_per_device=self.samples_per_device,
        device_idx=jax.process_index(),
        baseline_losses=self.baseline_losses if self.use_baseline_losses else None)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        bc_grad=mean_bc_grads,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(
        metrics, use_jnp=jax_utils.in_jit(), key=next(rng))
    if with_summary:
      # metrics["sample||delta_loss_sample"] = summary.sample_value(
      #     key, jnp.abs(delta_loss))
      # metrics["mean||delta_loss_mean"] = jnp.abs(delta_loss)
      metrics["sample||baseline_loss"] = b_loss
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics


@functools.partial(jax.pmap, axis_name="dev")
def _pmap_reduce(vals):
  return jax.lax.pmean(vals, axis_name="dev")


@jax.pmap
def vec_key_split(key):
  key1, key2 = jax.random.split(key)
  return key1, key2


@gin.configurable
class TruncatedPESPMAP(TruncatedPES):
  """GradientEstimator for computing PES gradient estimates leveraging pmap.

  See TruncatedPES documentation for information on PES. This estimator
  additionally makes use of multiple TPU devices via jax's pmap.
  """

  def __init__(self,
               *args,
               num_devices=8,
               replicate_data_across_devices=False,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.num_devices = num_devices
    self.replicate_data_across_devices = replicate_data_across_devices
    if len(jax.local_devices()) != self.num_devices:
      raise ValueError("Mismatch in device count!"
                       f" Found: {jax.local_devices()}."
                       f" Expected {num_devices} devices.")

    self.pmap_init_step_state = jax.pmap(
        self.truncated_step.init_step_state, in_axes=(None, None, 0))

    self.pmap_compute_pes_grad = jax.pmap(
        functools.partial(compute_pes_grad, std=self.std))

    self.pmap_vector_sample_perturbations = jax.pmap(
        functools.partial(
            common.vector_sample_perturbations,
            std=self.std,
            num_samples=self.truncated_step.num_tasks),
        in_axes=(None, 0),
    )

  @functools.partial(
      jax.pmap,
      in_axes=(None, 0, 0, 0, 0, None, None),
      static_broadcasted_argnums=(
          0,
          6,
      ))
  def pmap_unroll_next_state(self, vec_theta, key, state, datas, outer_state,
                             with_summary):
    theta_is_vector = True
    key1, key2 = jax.random.split(key)
    override_num_steps = None
    (p_state, p_ys), m = common.truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking
        self.truncated_step,
        self.steps_per_jit,
        theta_is_vector,
        vec_theta,
        key1,
        state,
        datas,
        outer_state,
        override_num_steps,
        with_summary=with_summary,
        sample_rng_key=key2)
    return (p_state, p_ys), m

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    theta = worker_weights.theta

    keys = jax.random.split(key, self.num_devices)
    # Note this doesn't use sampled theta for the first init.
    # I believe this is fine most of the time.
    # TODO(lmetz) consider init-ing at an is_done state instead.
    pos_unroll_state = self.pmap_init_step_state(worker_weights.theta,
                                                 worker_weights.outer_state,
                                                 keys)
    neg_unroll_state = pos_unroll_state

    accumulator = jax.tree_util.tree_map(
        lambda x: jnp.zeros([self.truncated_step.num_tasks] + list(x.shape)),
        theta)
    accumulator = flax_jax_utils.replicate(accumulator)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    vec_key = jax.random.split(key, self.num_devices)

    theta = worker_weights.theta

    vec_key1, vec_key = vec_key_split(vec_key)
    vec_pos, vec_p_theta, vec_n_theta = self.pmap_vector_sample_perturbations(
        theta, vec_key1)

    p_yses = []
    n_yses = []
    metrics = []

    def get_batch():
      """Get batch with leading dims [num_devices, steps_per_jit, num_tasks]."""
      if self.replicate_data_across_devices:
        b = self.truncated_step.get_batch(self.steps_per_jit)
        return flax_jax_utils.replicate(b)
      else:
        # Use different data across the devices
        batches = [
            self.truncated_step.get_batch(self.steps_per_jit)
            for _ in range(self.num_devices)
        ]
        return tree_utils.tree_zip_onp(batches)

    for _ in range(self.trunc_length // self.steps_per_jit):
      datas = get_batch()

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

      vec_key1, vec_key = vec_key_split(vec_key)
      (p_state, p_ys), m = self.pmap_unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
          vec_p_theta, vec_key, p_state, datas, worker_weights.outer_state,
          with_summary)
      metrics.append(m)

      p_yses.append(p_ys)
      (n_state, n_ys), _ = self.pmap_unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
          vec_n_theta, vec_key, n_state, datas, worker_weights.outer_state,
          False)
      n_yses.append(n_ys)

    loss, es_grad, new_accumulator, p_ys, delta_loss = self.pmap_compute_pes_grad(
        p_yses, n_yses, accumulator, vec_pos)

    es_grad, loss = flax_jax_utils.unreplicate(_pmap_reduce((es_grad, loss)))

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(metrics)
    if with_summary:
      metrics["sample||delta_loss_sample"] = summary.sample_value(
          key, jnp.abs(delta_loss))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_loss)
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
