# AdaFisher-inspired MLP Learned Optimizer
#
# Combines the adafac_mlp_lopt framework with AdaFisher's curvature signal:
#   - H_D (activation covariance diagonal) approximated from gradient column norms
#   - S_D (gradient covariance diagonal) approximated from gradient row norms
#   - Fisher preconditioner  F̃ = S_D * H_D + λ  (λ is meta-learned)
#   - Fisher-preconditioned gradient and momentum fed as MLP features


import functools
from typing import Any, Optional

import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import numpy as onp

PRNGKey = jnp.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def tanh_embedding(x):
    f32 = jnp.float32
    def one_freq(timescale):
        return jnp.tanh(x / f32(timescale) - 1.0)
    timescales = jnp.asarray(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
        dtype=jnp.float32)
    return jax.vmap(one_freq)(timescales)


def decay_to_param(x):
    return jnp.log(1 - x) / 10.


def param_to_decay(x):
    return 1 - jnp.exp(x * 10.)


# ─────────────────────────────────────────────────────────────────────────────
# Fisher factor helpers  (plain arrays only — no custom pytree nodes)
#
# h_d  = col-norm²  of gradient  (input-side curvature,  like AdaFisher's H_D)
# s_d  = row-norm²  of gradient  (output-side curvature, like AdaFisher's S_D)
#
# Both are shape (*param_shape, n_f) so they can be concatenated directly
# with other per-element features inside the MLP.
# ─────────────────────────────────────────────────────────────────────────────

def _init_fisher_factor(p: jnp.ndarray, n_f: int) -> jnp.ndarray:
    """All-ones initialisation  →  neutral (identity) preconditioner."""
    return jnp.ones(p.shape + (n_f,), dtype=jnp.float32)


# def _compute_h_d(g: jnp.ndarray, n_f: int) -> jnp.ndarray:
#     """Column squared-norms of g, normalised to [0,1], tiled to (*g.shape, n_f)."""
#     if g.ndim == 0:
#         v = jnp.ones((n_f,), dtype=jnp.float32)
#         return v  # scalar param: curvature is uniform
#     if g.ndim == 1:
#         sq = jnp.square(g)
#         sq = sq / (jnp.max(sq) + 1e-10)
#         return jnp.tile(sq[:, None], [1, n_f])
#     rows  = g.shape[0]
#     g2d   = g.reshape(rows, -1)
#     h_vec = jnp.sum(jnp.square(g2d), axis=0)          # (cols,)
#     h_vec = h_vec / (jnp.max(h_vec) + 1e-10)
#     exp   = jnp.tile(h_vec[None, :, None], [rows, 1, n_f])
#     return exp.reshape(g.shape + (n_f,))


def _compute_h_d(g: jnp.ndarray, n_f: int) -> jnp.ndarray:
    if g.ndim <= 1:
        sq = jnp.square(g) if g.ndim == 1 else jnp.ones((1,))
        # No normalisation — preserve absolute scale
        return jnp.tile(sq[:, None], [1, n_f]) if g.ndim == 1 else sq.reshape((n_f,))
    rows = g.shape[0]
    g2d  = g.reshape(rows, -1)
    h_vec = jnp.sum(jnp.square(g2d), axis=0)   # raw col norms²
    exp   = jnp.tile(h_vec[None, :, None], [rows, 1, n_f])
    return exp.reshape(g.shape + (n_f,))


# def _compute_s_d(g: jnp.ndarray, n_f: int) -> jnp.ndarray:
#     """Row squared-norms of g, mean-normalised and tiled to (*g.shape, n_f)."""
#     if g.ndim == 0:
#         return jnp.ones((n_f,), dtype=jnp.float32)
#     if g.ndim == 1:
#         sq = jnp.square(g)
#         sq = sq / (jnp.mean(sq) + 1e-10)   # << mean-normalize (consistent)
#         return jnp.tile(sq[:, None], [1, n_f])
#     rows  = g.shape[0]
#     g2d   = g.reshape(rows, -1)
#     s_vec = jnp.sum(jnp.square(g2d), axis=1)          # (rows,)
#     s_vec = s_vec / (jnp.mean(s_vec) + 1e-10)         # << mean
#     exp   = jnp.tile(s_vec[:, None, None], [1, g2d.shape[1], n_f])
#     return exp.reshape(g.shape + (n_f,))

def _compute_s_d(g: jnp.ndarray, n_f: int) -> jnp.ndarray:
    """Row squared-norms of g, tiled to (*g.shape, n_f)."""
    if g.ndim == 0:
        return jnp.ones((n_f,), dtype=jnp.float32)
    if g.ndim == 1:
        sq = jnp.square(g)
        return jnp.tile(sq[:, None], [1, n_f])
    rows  = g.shape[0]
    g2d   = g.reshape(rows, -1)
    s_vec = jnp.sum(jnp.square(g2d), axis=1)          # (rows,)
    exp   = jnp.tile(s_vec[:, None, None], [1, g2d.shape[1], n_f])
    return exp.reshape(g.shape + (n_f,))


def _update_fisher_h_d(h_d, g, fisher_decay):
    """EMA update for h_d.  All arguments are plain jnp arrays."""
    return fisher_decay * h_d + (1.0 - fisher_decay) * _compute_h_d(g, fisher_decay.shape[0])


def _update_fisher_s_d(s_d, g, fisher_decay):
    """EMA update for s_d.  All arguments are plain jnp arrays."""
    return fisher_decay * s_d + (1.0 - fisher_decay) * _compute_s_d(g, fisher_decay.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer state
# ─────────────────────────────────────────────────────────────────────────────

@flax.struct.dataclass
class AdaFisherMLPLOptState:
    params:      Any
    state:       Any
    mom_rolling: common.MomAccumulator
    # Plain pytrees mirroring params.  Each leaf: (*param_shape, n_f).
    fisher_h_d:  Any
    fisher_s_d:  Any
    num_steps:   jnp.ndarray
    iteration:   jnp.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Learned optimizer
# ─────────────────────────────────────────────────────────────────────────────

@gin.configurable
class AdaFisherMLPLOpt(lopt_base.LearnedOptimizer):
    """
    MLP learned optimizer with AdaFisher-style diagonal Fisher preconditioner.

    Meta-learned parameters (theta):
      nn               : MLP weights
      momentum_decays  : offsets on initial_momentum_decays (in param space)
      fisher_decays    : offsets on initial_fisher_decays
      log_lambda       : log(λ), λ = exp(log_lambda) > 0   init ≈ log(1e-3)

    Per-element MLP input features:
      g, p             gradient / parameter value            (*, 1)
      m                momentum accumulators                  (*, n_m)
      h_d, s_d         Fisher factors                         (*, n_f)
      f_tilde          s_d * h_d + λ                          (*, n_f)
      rsqrt_f          1 / sqrt(f_tilde)                      (*, n_f)
      g * rsqrt_f      preconditioned gradient                (*, n_f)
      mean(m) * rsqrt_f preconditioned momentum               (*, n_f)
      training_step_feature   global step embedding           (11,)

    Update:
      step  = direction * exp(magnitude * exp_mult) * step_mult
      new_p = p - step
    """

    def __init__(
            self,
            exp_mult: float = 0.001,
            step_mult: float = 0.001,
            hidden_size: int = 32,
            hidden_layers: int = 4,
            initial_momentum_decays=(0.9, 0.99, 0.999),
            initial_fisher_decays=(0.9, 0.99),
    ):
        super().__init__()
        self._exp_mult = exp_mult
        self._step_mult = step_mult
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._initial_momentum_decays = initial_momentum_decays
        self._initial_fisher_decays = initial_fisher_decays

        self._mod_init, self._mod_apply = hk.without_apply_rng(
            hk.transform(self._mod))

    # ── MLP forward ───────────────────────────────────────────────────────

    def _mod(self, global_feat, p, g, m, h_d, s_d, lam):
        did_reshape = False
        if not p.shape:
            p   = jnp.expand_dims(p,   0)
            g   = jnp.expand_dims(g,   0)
            m   = jnp.expand_dims(m,   0)
            h_d = jnp.expand_dims(h_d, 0)
            s_d = jnp.expand_dims(s_d, 0)
            did_reshape = True

        
#         f_tilde = jnp.maximum(s_d * h_d + lam, 1e-12)
#         rsqrt_f = lax.rsqrt(f_tilde + 1e-12)
#         m_mean  = jnp.mean(m, axis=-1, keepdims=True)
        f_tilde = s_d * h_d + lam
        #f_tilde = jnp.clip(f_tilde, 1e-8, 1e6)      # floor and ceiling
        rsqrt_f = lax.rsqrt(f_tilde + 1e-12)
        m_mean  = jnp.mean(m, axis=-1, keepdims=True)

#         inps = [
#             jnp.expand_dims(g, -1),
#             jnp.expand_dims(p, -1),
#             m,
#             h_d,
#             s_d,
#             f_tilde,
#             rsqrt_f,
#             jnp.expand_dims(g, -1) * rsqrt_f,   # preconditioned gradient
#             m_mean * rsqrt_f,                     # preconditioned momentum ← was dropped
#             jnp.log(h_d + 1e-12),                # keep log features as extra signal
#             jnp.log(s_d + 1e-12),
#         ]
        inps = [
            jnp.expand_dims(g, -1),           # raw gradient
            jnp.expand_dims(p, -1),           # parameter value
            m,                                 # momentum (n_m=3)
            jnp.expand_dims(g, -1) * rsqrt_f, # preconditioned gradient (n_f)
            m_mean * rsqrt_f,                  # preconditioned momentum (n_f)
            jnp.log(f_tilde + 1e-12),         # curvature magnitude (n_f)
        ]
        # = 1 + 1 + 3 + 2 + 2 + 2 = 11 features — comparable to adafac



        last_size = jnp.concatenate(inps, axis=-1).shape[-1]
        last_size += global_feat["training_step_feature"].shape[-1]
        weights, biases = [], []
        for wi, w_out in enumerate([self._hidden_size] * self._hidden_layers + [2]):
            stddev = 1.0 / onp.sqrt(last_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            weights.append(hk.get_parameter(
                f"w{wi}", shape=(last_size, w_out), dtype=jnp.float32, init=w_init))
            biases.append(hk.get_parameter(
                f"b{wi}", shape=(w_out,), dtype=jnp.float32, init=jnp.zeros))
            last_size = w_out

        inp_stack = jnp.concatenate(inps, axis=-1)
        axis      = list(range(len(p.shape)))
        #inp_stack = second_moment_normalizer(inp_stack, axis=axis)
        inp_stack = second_moment_normalizer(inp_stack, axis=[-1])

        tsf       = global_feat["training_step_feature"]
        tsf_tiled = jnp.tile(
            jnp.reshape(tsf, [1] * len(axis) + [tsf.shape[-1]]),
            list(p.shape) + [1])
        net = jnp.concatenate([inp_stack, tsf_tiled], axis=-1)

        for wi, (w, b) in enumerate(zip(weights, biases)):
            net = net @ w + jnp.broadcast_to(b, list(net.shape[:-1]) + [w.shape[-1]])
            if wi != len(weights) - 1:
                net = jax.nn.relu(net)

        direction = net[..., 0]
        magnitude = net[..., 1]
        step      = direction * jnp.exp(magnitude * self._exp_mult) * self._step_mult
        step      = step.reshape(p.shape)
        new_p     = p - step

        if did_reshape:
            new_p = jnp.squeeze(new_p, 0)

        avg_step = jnp.mean(jnp.abs(step))
        summary.summary("adafisher_mlp_lopt/avg_step_size", avg_step)
        summary.summary("adafisher_mlp_lopt/avg_step_size_hist",
                        avg_step, aggregation="collect")
        summary.summary("adafisher_mlp_lopt/direction/mean_abs",
                        jnp.mean(jnp.abs(direction)))
        summary.summary("adafisher_mlp_lopt/magnitude/mean_abs",
                        jnp.mean(jnp.abs(magnitude)))
        summary.summary("adafisher_mlp_lopt/f_tilde/mean", jnp.mean(f_tilde))
        summary.summary("adafisher_mlp_lopt/lambda", lam)
        summary.summary("adafisher_mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

        return new_p

    # ── Meta-init ─────────────────────────────────────────────────────────

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        training_step_feature = tanh_embedding(1)
        global_features = {
            "iterations": 0, "num_steps": 10,
            "training_step_feature": training_step_feature,
        }
        r, c = 10, 10
        n_m  = len(self._initial_momentum_decays)
        n_f  = len(self._initial_fisher_decays)

        p   = jnp.ones([r, c])
        g   = jnp.ones([r, c])
        m   = jnp.ones([r, c, n_m])
        h_d = jnp.ones([r, c, n_f])
        s_d = jnp.ones([r, c, n_f])
        lam = jnp.asarray(1e-5)

        mod_theta = self._mod_init(key, global_features, p, g, m, h_d, s_d, lam)

        return hk.data_structures.to_haiku_dict({
            "momentum_decays": jnp.zeros([n_m]),
            "fisher_decays":   jnp.zeros([n_f]),
            "log_lambda": jnp.full((n_f,), -2.6, dtype=jnp.float32),  # exp(-4.6) ≈ 0.01
            #"log_lambda":      jnp.asarray(-6.9, dtype=jnp.float32),
            "nn":              mod_theta,
        })
    
    

    # ── Inner optimizer factory ───────────────────────────────────────────

    def opt_fn(self,
               theta: lopt_base.MetaParams,
               is_training: Optional[bool] = False) -> opt_base.Optimizer:

        mod_apply = self._mod_apply
        parent    = self

        class _Opt(opt_base.Optimizer):

            def __init__(self, theta):
                self.theta = theta

            def _get_decays(self):
                mom_decay = param_to_decay(
                    decay_to_param(jnp.asarray(parent._initial_momentum_decays))
                    + self.theta["momentum_decays"])
                fisher_decay = param_to_decay(
                    decay_to_param(jnp.asarray(parent._initial_fisher_decays))
                    + self.theta["fisher_decays"])
                return mom_decay, fisher_decay

  
            def init(
                    self,
                    params:      opt_base.Params,
                    model_state: Optional[opt_base.ModelState] = None,
                    num_steps:   Optional[int] = None,
                    key:         Optional[PRNGKey] = None,
                    init_grad:   Optional[opt_base.Gradient] = None,   # ← add this
            ) -> AdaFisherMLPLOptState:
                if num_steps is None:
                    raise ValueError("AdaFisherMLPLOpt requires num_steps at init time.")

                mom_decay, fisher_decay = self._get_decays()
                mom_roll = common.vec_rolling_mom(mom_decay)
                n_f = len(parent._initial_fisher_decays)

                if init_grad is not None:
                    # Warm-start: compute real curvature estimates from the first gradient
                    fisher_h_d = jax.tree_util.tree_map(
                        lambda g: _compute_h_d(g, n_f), init_grad)
                    fisher_s_d = jax.tree_util.tree_map(
                        lambda g: _compute_s_d(g, n_f), init_grad)
                else:
                    # Cold fallback: all-ones (original behaviour)
                    fisher_h_d = jax.tree_util.tree_map(
                        lambda p: _init_fisher_factor(p, n_f), params)
                    fisher_s_d = jax.tree_util.tree_map(
                        lambda p: _init_fisher_factor(p, n_f), params)

                return AdaFisherMLPLOptState(
                    params=params,
                    state=model_state,
                    mom_rolling=mom_roll.init(params),
                    fisher_h_d=fisher_h_d,
                    fisher_s_d=fisher_s_d,
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    num_steps=jnp.asarray(num_steps))

            def update(
                    self,  # pytype: disable=signature-mismatch
                    opt_state:   AdaFisherMLPLOptState,
                    grad:        opt_base.Gradient,
                    loss:        jnp.ndarray,
                    model_state: Optional[opt_base.ModelState] = None,
                    is_valid:    bool = False,
                    key:         Optional[PRNGKey] = None,
            ) -> AdaFisherMLPLOptState:

                mom_decay, fisher_decay = self._get_decays()
                mom_roll = common.vec_rolling_mom(mom_decay)

                next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grad)

                # tree_map over plain arrays: lambda receives two plain arrays,
                # never a FisherFactorState struct — this is the core fix.
                next_h_d = jax.tree_util.tree_map(
                    lambda h, g: _update_fisher_h_d(h, g, fisher_decay),
                    opt_state.fisher_h_d, grad)

                next_s_d = jax.tree_util.tree_map(
                    lambda s, g: _update_fisher_s_d(s, g, fisher_decay),
                    opt_state.fisher_s_d, grad)

                #lam = jnp.exp(self.theta["log_lambda"])
                clamped_log_lambda = jnp.clip(self.theta["log_lambda"], a_min=-12.0, a_max=3.0)
                lam = jnp.exp(clamped_log_lambda) 

                training_step_feature = tanh_embedding(opt_state.iteration)
                global_features = {
                    "iterations":            opt_state.iteration,
                    "num_steps":             opt_state.num_steps,
                    "training_step_feature": training_step_feature,
                }

                fun = functools.partial(mod_apply, self.theta["nn"], global_features)

                next_params = jax.tree_util.tree_map(
                    lambda p, g, m, h, s: fun(p, g, m, h, s, lam),
                    opt_state.params,
                    grad,
                    next_mom_rolling.m,
                    next_h_d,
                    next_s_d)

                next_opt_state = AdaFisherMLPLOptState(
                    params=next_params,
                    state=model_state,
                    mom_rolling=next_mom_rolling,
                    fisher_h_d=next_h_d,
                    fisher_s_d=next_s_d,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps)

                return tree_utils.match_type(next_opt_state, opt_state)

        return _Opt(theta)