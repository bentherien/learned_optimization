# coding=utf-8
# AdaFac + AdaFisher augmentation — Version: Per-Channel Preconditioned Momentum
#
# Feature block: h_d, s_d, f_tilde, rsqrt_f, preconditioned gradient,
# and per-channel momentum preconditioned by Fisher (outer product).

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
# Shared utilities
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
# Fisher factor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_fisher_factor(p: jnp.ndarray, n_f: int) -> jnp.ndarray:
    return jnp.ones(p.shape + (n_f,), dtype=jnp.float32)


def _compute_h_d(g: jnp.ndarray, n_f: int) -> jnp.ndarray:
    if g.ndim == 0:
        return jnp.ones((n_f,), dtype=jnp.float32)
    if g.ndim == 1:
        sq = jnp.square(g)
        return jnp.tile(sq[:, None], [1, n_f])
    rows  = g.shape[0]
    g2d   = g.reshape(rows, -1)
    h_vec = jnp.sum(jnp.square(g2d), axis=0)
    exp   = jnp.tile(h_vec[None, :, None], [rows, 1, n_f])
    return exp.reshape(g.shape + (n_f,))


def _compute_s_d(g: jnp.ndarray, n_f: int) -> jnp.ndarray:
    if g.ndim == 0:
        return jnp.ones((n_f,), dtype=jnp.float32)
    if g.ndim == 1:
        sq = jnp.square(g)
        return jnp.tile(sq[:, None], [1, n_f])
    rows  = g.shape[0]
    g2d   = g.reshape(rows, -1)
    s_vec = jnp.sum(jnp.square(g2d), axis=1)
    exp   = jnp.tile(s_vec[:, None, None], [1, g2d.shape[1], n_f])
    return exp.reshape(g.shape + (n_f,))


def _update_fisher_h_d(h_d, g, fisher_decay):
    return fisher_decay * h_d + (1.0 - fisher_decay) * _compute_h_d(g, fisher_decay.shape[0])


def _update_fisher_s_d(s_d, g, fisher_decay):
    return fisher_decay * s_d + (1.0 - fisher_decay) * _compute_s_d(g, fisher_decay.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@flax.struct.dataclass
class AdaFisherPerChannelMLPLOptState:
    params:               Any
    state:                Any
    mom_rolling:          common.MomAccumulator
    rms_rolling:          common.RMSAccumulator
    fac_rolling_features: common.FactoredAccum
    num_steps:            jnp.ndarray
    iteration:            jnp.ndarray
    fisher_h_d:           Any   # (*param_shape, n_f)
    fisher_s_d:           Any   # (*param_shape, n_f)


# ─────────────────────────────────────────────────────────────────────────────
# Learned optimizer
# ─────────────────────────────────────────────────────────────────────────────

@gin.configurable
class AdaFisherPerChannelMLPLOpt(lopt_base.LearnedOptimizer):
    """
    AdaFac MLP learned optimizer — Version: Per-Channel Preconditioned Momentum.

    Fisher features appended to AdaFac feature vector:
        h_d        col-norm² EMA                                   (*, n_f)
        s_d        row-norm² EMA                                   (*, n_f)
        f_tilde    s_d * h_d + λ                                   (*, n_f)
        rsqrt_f    1 / sqrt(f_tilde)                               (*, n_f)
        g*rsqrt_f  Fisher-preconditioned gradient                  (*, n_f)
        m_fisher   per-channel momentum × Fisher rsqrt outer prod  (*, n_m*n_f)
    """

    def __init__(
            self,
            exp_mult:                float = 0.001,
            step_mult:               float = 0.001,
            hidden_size:             int   = 4,
            hidden_layers:           int   = 2,
            initial_momentum_decays        = (0.9, 0.99, 0.999),
            initial_rms_decays             = (0.999,),
            initial_adafactor_decays       = (0.9, 0.99, 0.999),
            initial_fisher_decays          = (0.9, 0.99),
    ):
        super().__init__()
        self._exp_mult                = exp_mult
        self._step_mult               = step_mult
        self._hidden_size             = hidden_size
        self._hidden_layers           = hidden_layers
        self._initial_momentum_decays = initial_momentum_decays
        self._initial_rms_decays      = initial_rms_decays
        self._initial_adafactor_decays= initial_adafactor_decays
        self._initial_fisher_decays   = initial_fisher_decays

        self._mod_init, self._mod_apply = hk.without_apply_rng(
            hk.transform(self._mod))

    def _mod(self, global_feat,
             p, g, m, rms, fac_g, fac_vec_col, fac_vec_row, fac_vec_v,
             h_d, s_d, lam):

        did_reshape = False
        if not p.shape:
            p           = jnp.expand_dims(p,   0)
            g           = jnp.expand_dims(g,   0)
            m           = jnp.expand_dims(m,   0)
            rms         = jnp.expand_dims(rms, 0)
            fac_g       = jnp.expand_dims(fac_g, 0)
            fac_vec_v   = jnp.expand_dims(fac_vec_v, 0)
            fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
            fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
            h_d         = jnp.expand_dims(h_d, 0)
            s_d         = jnp.expand_dims(s_d, 0)
            did_reshape = True

        # ── AdaFac feature block ─────────────────────────────────────────
        inps = []
        inps.append(jnp.expand_dims(g,   axis=-1))
        inps.append(jnp.expand_dims(p,   axis=-1))
        inps.append(m)
        inps.append(rms)
        rsqrt = lax.rsqrt(rms + 1e-6)
        inps.append(m * rsqrt)
        inps.append(rsqrt)
        inps.append(fac_g)

        factored_dims = common.factored_dims(g.shape)
        if factored_dims is not None:
            d1, d0 = factored_dims

            to_tile = [1] * (1 + len(g.shape))
            to_tile[d0] = g.shape[d0]
            row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

            to_tile = [1] * (1 + len(g.shape))
            to_tile[d1] = g.shape[d1]
            col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

            inps.append(row_feat)
            inps.append(col_feat)
            inps.append(lax.rsqrt(row_feat + 1e-8))
            inps.append(lax.rsqrt(col_feat + 1e-8))

            reduced_d1    = d1 - 1 if d1 > d0 else d1
            row_col_mean  = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)
            row_factor    = common.safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
            col_factor    = common.safe_rsqrt(fac_vec_col)
            fac_mom_mult  = (m
                             * jnp.expand_dims(row_factor, axis=d0)
                             * jnp.expand_dims(col_factor, axis=d1))
            inps.append(fac_mom_mult)
        else:
            inps.append(fac_vec_v)
            inps.append(fac_vec_v)
            inps.append(lax.rsqrt(fac_vec_v + 1e-8))
            inps.append(lax.rsqrt(fac_vec_v + 1e-8))
            fac_mom_mult = m * (fac_vec_v + 1e-6) ** -0.5
            inps.append(fac_mom_mult)

        # ── Per-Channel Preconditioned Momentum block ────────────────────
        n_m = m.shape[-1]
        n_f = h_d.shape[-1]

        f_tilde  = jnp.maximum(s_d * h_d + lam, 1e-12)   # (*, n_f)
        rsqrt_f  = lax.rsqrt(f_tilde + 1e-12)             # (*, n_f)

        inps.append(h_d)                                   # (*, n_f)
        inps.append(s_d)                                   # (*, n_f)
        inps.append(f_tilde)                               # (*, n_f)
        inps.append(rsqrt_f)                               # (*, n_f)
        inps.append(jnp.expand_dims(g, -1) * rsqrt_f)     # (*, n_f)

        m_exp     = m[..., :, None]                        # (*, n_m, 1)
        rsqrt_exp = rsqrt_f[..., None, :]                  # (*, 1,   n_f)
        m_fisher  = (m_exp * rsqrt_exp).reshape(p.shape + (n_m * n_f,))
        inps.append(m_fisher)                              # (*, n_m*n_f)

        # ── MLP ──────────────────────────────────────────────────────────
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
        inp_stack = second_moment_normalizer(inp_stack, axis=axis)

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
        summary.summary("adafac_fisher_perchannel/avg_step_size", avg_step)
        summary.summary("adafac_fisher_perchannel/avg_step_size_hist", avg_step, aggregation="collect")
        summary.summary("adafac_fisher_perchannel/direction/mean_abs",  jnp.mean(jnp.abs(direction)))
        summary.summary("adafac_fisher_perchannel/magnitude/mean_abs",  jnp.mean(jnp.abs(magnitude)))
        summary.summary("adafac_fisher_perchannel/magnitude/mean",      jnp.mean(magnitude))
        summary.summary("adafac_fisher_perchannel/grad/mean_abs",       jnp.mean(jnp.abs(g)))
        summary.summary("adafac_fisher_perchannel/f_tilde/mean",        jnp.mean(f_tilde))
        summary.summary("adafac_fisher_perchannel/lambda",              lam)

        return new_p

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        training_step_feature = tanh_embedding(1)
        global_features = {
            "iterations": 0, "num_steps": 10,
            "training_step_feature": training_step_feature,
        }
        r, c  = 10, 10
        n_m   = len(self._initial_momentum_decays)
        n_rms = len(self._initial_rms_decays)
        n_fac = len(self._initial_adafactor_decays)
        n_f   = len(self._initial_fisher_decays)

        p           = jnp.ones([r, c])
        g           = jnp.ones([r, c])
        m           = jnp.ones([r, c, n_m])
        rms         = jnp.ones([r, c, n_rms])
        fac_g       = jnp.ones([r, c, n_fac])
        fac_vec_row = jnp.ones([r, n_fac])
        fac_vec_col = jnp.ones([c, n_fac])
        fac_vec_v   = jnp.ones([n_fac])
        h_d = jnp.ones([r, c, n_f])
        s_d = jnp.ones([r, c, n_f])
        lam = jnp.asarray(1e-3)

        mod_theta = self._mod_init(
            key, global_features,
            p, g, m, rms, fac_g, fac_vec_col, fac_vec_row, fac_vec_v,
            h_d, s_d, lam)

        return hk.data_structures.to_haiku_dict({
            "momentum_decays":  jnp.zeros([n_m]),
            "rms_decays":       jnp.zeros([n_rms]),
            "adafactor_decays": jnp.zeros([n_fac]),
            "fisher_decays":    jnp.zeros([n_f]),
            "log_lambda":       jnp.asarray(-6.9, dtype=jnp.float32),
            "nn":               mod_theta,
        })

    def opt_fn(self,
               theta: lopt_base.MetaParams,
               is_training: Optional[bool] = False) -> opt_base.Optimizer:

        mod_apply = self._mod_apply
        parent    = self

        class _Opt(opt_base.Optimizer):

            def __init__(self, theta):
                self.theta = theta

            def _get_rolling(self):
                mom_decay = param_to_decay(
                    decay_to_param(jnp.asarray(parent._initial_momentum_decays))
                    + self.theta["momentum_decays"])
                mom_roll = common.vec_rolling_mom(mom_decay)

                rms_decay = param_to_decay(
                    decay_to_param(jnp.asarray(parent._initial_rms_decays))
                    + self.theta["rms_decays"])
                rms_roll = common.vec_rolling_rms(rms_decay)

                adafactor_decay = param_to_decay(
                    decay_to_param(jnp.asarray(parent._initial_adafactor_decays))
                    + self.theta["adafactor_decays"])
                fac_vec_roll = common.vec_factored_rolling(adafactor_decay)
                return mom_roll, rms_roll, fac_vec_roll

            def _get_fisher_decay(self):
                return param_to_decay(
                    decay_to_param(jnp.asarray(parent._initial_fisher_decays))
                    + self.theta["fisher_decays"])

            def init(
                    self,
                    params:      opt_base.Params,
                    model_state: Optional[opt_base.ModelState] = None,
                    num_steps:   Optional[int] = None,
                    key:         Optional[PRNGKey] = None,
            ) -> AdaFisherPerChannelMLPLOptState:
                if num_steps is None:
                    raise ValueError("AdaFisherPerChannelMLPLOpt requires num_steps at init time.")

                mom_roll, rms_roll, fac_vec_roll = self._get_rolling()
                n_f = len(parent._initial_fisher_decays)

                return AdaFisherPerChannelMLPLOptState(
                    params=params,
                    state=model_state,
                    mom_rolling=mom_roll.init(params),
                    rms_rolling=rms_roll.init(params),
                    fac_rolling_features=fac_vec_roll.init(params),
                    fisher_h_d=jax.tree_util.tree_map(
                        lambda p: _init_fisher_factor(p, n_f), params),
                    fisher_s_d=jax.tree_util.tree_map(
                        lambda p: _init_fisher_factor(p, n_f), params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    num_steps=jnp.asarray(num_steps))

            def update(
                    self,
                    opt_state:   AdaFisherPerChannelMLPLOptState,
                    grad:        opt_base.Gradient,
                    loss:        jnp.ndarray,
                    model_state: Optional[opt_base.ModelState] = None,
                    is_valid:    bool = False,
                    key:         Optional[PRNGKey] = None,
            ) -> AdaFisherPerChannelMLPLOptState:

                mom_roll, rms_roll, fac_vec_roll = self._get_rolling()
                next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grad)
                next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grad)
                next_fac_rolling_features, fac_g = fac_vec_roll.update(
                    opt_state.fac_rolling_features, grad)

                fisher_decay = self._get_fisher_decay()
                next_h_d = jax.tree_util.tree_map(
                    lambda h, g: _update_fisher_h_d(h, g, fisher_decay),
                    opt_state.fisher_h_d, grad)
                next_s_d = jax.tree_util.tree_map(
                    lambda s, g: _update_fisher_s_d(s, g, fisher_decay),
                    opt_state.fisher_s_d, grad)

                t  = (opt_state.iteration + 1).astype(jnp.float32)
                bc = 1.0 - fisher_decay ** t
                h_d_for_mlp = jax.tree_util.tree_map(
                    lambda h: h / (bc + 1e-12), next_h_d)
                s_d_for_mlp = jax.tree_util.tree_map(
                    lambda s: s / (bc + 1e-12), next_s_d)

                lam = jnp.exp(jnp.clip(self.theta["log_lambda"], -12.0, 3.0))

                training_step_feature = tanh_embedding(opt_state.iteration)
                global_features = {
                    "iterations":            opt_state.iteration,
                    "num_steps":             opt_state.num_steps,
                    "training_step_feature": training_step_feature,
                }

                fun = functools.partial(mod_apply, self.theta["nn"], global_features)

                next_params = jax.tree_util.tree_map(
                    lambda p, g, m, rms, fg, vc, vr, vd, h, s:
                        fun(p, g, m, rms, fg, vc, vr, vd, h, s, lam),
                    opt_state.params,
                    grad,
                    next_mom_rolling.m,
                    next_rms_rolling.rms,
                    fac_g,
                    next_fac_rolling_features.v_col,
                    next_fac_rolling_features.v_row,
                    next_fac_rolling_features.v_diag,
                    h_d_for_mlp,
                    s_d_for_mlp)

                next_opt_state = AdaFisherPerChannelMLPLOptState(
                    params=next_params,
                    state=model_state,
                    mom_rolling=next_mom_rolling,
                    rms_rolling=next_rms_rolling,
                    fac_rolling_features=next_fac_rolling_features,
                    fisher_h_d=next_h_d,
                    fisher_s_d=next_s_d,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps)

                return tree_utils.match_type(next_opt_state, opt_state)

        return _Opt(theta)
