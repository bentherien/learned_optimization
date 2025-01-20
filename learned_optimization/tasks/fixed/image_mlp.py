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

"""Tasks based on MLP."""
# pylint: disable=invalid-name

import functools
from collections.abc import Iterable
from typing import Any, Mapping, Tuple, Callable, Optional

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image
import numpy as onp

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray



class MLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      log_activations: bool = False,
      name: Optional[str] = None,
  ):
    """Constructs an MLP.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    self.log_activations = log_activations
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      layers.append(hk.Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index))
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jax.Array:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)

      if self.log_activations:
        hk.set_state("layer_%d_pre-act_l1" % i, jnp.mean(jnp.abs(out)))
        hk.set_state("layer_%d_pre-act" % i, out)

      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

        if self.log_activations:
          hk.set_state("layer_%d_act_l1" % i, jnp.mean(jnp.abs(out)))
          hk.set_state("layer_%d_act" % i, out)
      else:
        if self.log_activations:
          hk.set_state("layer_%d_logits_l1" % i, jnp.mean(jnp.abs(out * self.output_mul)))
          hk.set_state("layer_%d_logits" % i, out * self.output_mul)

    return out

  def reverse(
      self,
      activate_final: Optional[bool] = None,
      name: Optional[str] = None,
  ) -> "MLP":
    """Returns a new MLP which is the layer-wise reverse of this MLP.

    NOTE: Since computing the reverse of an MLP requires knowing the input size
    of each linear layer this method will fail if the module has not been called
    at least once.

    The contract of reverse is that the reversed module will accept the output
    of the parent module as input and produce an output which is the input size
    of the parent.

    >>> mlp = hk.nets.MLP([1, 2, 3])
    >>> mlp_in = jnp.ones([1, 2])
    >>> y = mlp(mlp_in)
    >>> rev = mlp.reverse()
    >>> rev_mlp_out = rev(y)
    >>> mlp_in.shape == rev_mlp_out.shape
    True

    Args:
      activate_final: Whether the final layer of the MLP should be activated.
      name: Optional name for the new module. The default name will be the name
        of the current module prefixed with ``"reversed_"``.

    Returns:
      An MLP instance which is the reverse of the current instance. Note these
      instances do not share weights and, apart from being symmetric to each
      other, are not coupled in any way.
    """

    if activate_final is None:
      activate_final = self.activate_final
    if name is None:
      name = self.name + "_reversed"

    output_sizes = tuple(
        layer.input_size
        for layer in reversed(self.layers)
        if layer.input_size is not None
    )
    if len(output_sizes) != len(self.layers):
      raise ValueError("You cannot reverse an MLP until it has been called.")
    return MLP(
        output_sizes=output_sizes,
        w_init=self.w_init,
        b_init=self.b_init,
        with_bias=self.with_bias,
        activation=self.activation,
        activate_final=activate_final,
        name=name,
    )



class _MLPImageTask(base.Task):
  """MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0,
               log_activations=False):
    super().__init__()
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.datasets = datasets

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return MLP(
          sizes, activation=act_fn, log_activations=log_activations)(
              inp, dropout_rate=dropout_rate, rng=hk.next_rng_key())

    self._mod = hk.transform(_forward)

  def init(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def loss_and_accuracy(self, params: Params, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])[0]
    
    # Calculate the loss as before
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(vec_loss)
    
    # Calculate the accuracy
    predictions = jnp.argmax(logits, axis=-1)
    actual = data["label"]
    correct_predictions = predictions == actual
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return loss, accuracy

  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable
def ImageMLP_Cifar10BW8_Relu32():
  """A 1 hidden layer, 32 unit MLP for 8x8 black and white cifar10."""
  datasets = image.cifar10_datasets(
      batch_size=128, image_size=(8, 8), convert_to_black_and_white=True)
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist_Relu128x128():
  """A 2 hidden layer, 128 hidden unit MLP designed for fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def ImageMLP_FashionMnist8_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(8, 8))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist16_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(16, 16))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist32_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(32, 32))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_Cifar10_8_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 cifar10."""
  datasets = image.cifar10_datasets(batch_size=128, image_size=(8, 8))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_Imagenet16_Relu256x256x256():
  """A 3 hidden layer MLP trained on 16x16 resized imagenet."""
  datasets = image.imagenet16_datasets(batch_size=128)
  return _MLPImageTask(datasets, [256, 256, 256])


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar100_128x128x128_Relu():
  datasets = image.cifar100_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Tanh_bs64():
  datasets = image.cifar10_datasets(batch_size=64)
  return _MLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Tanh_bs128():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Tanh_bs256():
  datasets = image.cifar10_datasets(batch_size=256)
  return _MLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Mnist_128x128x128_Relu():
  datasets = image.mnist_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar10_256x256_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [256, 256])


@gin.configurable
def ImageMLP_Cifar10_256x256_Relu_BS32():
  datasets = image.cifar10_datasets(batch_size=32)
  return _MLPImageTask(datasets, [256, 256])


@gin.configurable
def ImageMLP_Cifar10_1024x1024_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [1024, 1024])


@gin.configurable
def ImageMLP_Cifar10_4096x4096_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [4096, 4096])


@gin.configurable
def ImageMLP_Cifar10_8192x8192_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [8192, 8192])


@gin.configurable
def ImageMLP_Cifar10_16384x16384_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [16384, 16384])


class _MLPImageTaskMSE(_MLPImageTask):
  """Image model with a Mean squared error loss."""

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    return jnp.mean(jnp.square(logits - labels))

  def normalizer(self, loss):
    maxval = 1.0
    loss = jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)
    return jnp.minimum(loss, 1.0) * 10


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar10_128x128_Dropout05_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128], dropout_rate=0.5)


@gin.configurable
def ImageMLP_Cifar10_128x128_Dropout08_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128], dropout_rate=0.8)


@gin.configurable
def ImageMLP_Cifar10_128x128_Dropout02_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128], dropout_rate=0.2)


class _MLPImageTaskNorm(base.Task):
  """MLP based image task with layer norm."""

  def __init__(
      self,
      datasets,  # pylint: disable=super-init-not-called
      hidden_sizes,
      norm_type,
      act_fn=jax.nn.relu):
    self.datasets = datasets
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]

    def _forward(inp):
      net = jnp.reshape(inp, [inp.shape[0], -1])

      for i, h in enumerate(sizes):
        net = hk.Linear(h)(net)
        if i != (len(sizes) - 1):
          if norm_type == "layer_norm":
            net = hk.LayerNorm(
                axis=1, create_scale=True, create_offset=True)(
                    net)
          elif norm_type == "batch_norm":
            net = hk.BatchNorm(
                create_scale=True, create_offset=True, decay_rate=0.9)(
                    net, is_training=True)
          else:
            raise ValueError(f"No norm {norm_type} implemented!")
          net = act_fn(net)
      return net

    # Batchnorm has state -- though we don't use it here
    # (we are using training mode only for this loss.)
    self._mod = hk.transform_with_state(_forward)

  def init_with_state(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._mod.init(key, batch["image"])
    return params, state

  def loss_with_state(self, params: Params, state: ModelState, key: PRNGKey,
                      data: Any) -> Tuple[jnp.ndarray, ModelState]:
    num_classes = self.datasets.extra_info["num_classes"]
    logits, state = self._mod.apply(params, state, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss), state

  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Any) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    loss, state = self.loss_with_state(params, state, key, data)
    return loss, state, {}

  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_LayerNorm_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(datasets, [128, 128, 128], norm_type="layer_norm")


@gin.configurable
def ImageMLP_Cifar10_128x128x128_LayerNorm_Tanh():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(
      datasets, [128, 128, 128], norm_type="layer_norm", act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_BatchNorm_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(datasets, [128, 128, 128], norm_type="batch_norm")


@gin.configurable
def ImageMLP_Cifar10_128x128x128_BatchNorm_Tanh():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(
      datasets, [128, 128, 128], norm_type="batch_norm", act_fn=jnp.tanh)
