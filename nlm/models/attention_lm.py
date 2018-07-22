
"""Self-attention based language model.

Like transformer.py, but no encoder

decoder: [Self-Attention, Feed-forward] x n

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
import copy
import time
from six.moves import xrange  # pylint: disable=redefined-builtin

from nlm.models import common_attention
from nlm.models import common_layers
from nlm.utils import beam_search
from nlm.utils import expert_utils as eu
from . import modalities

import tensorflow as tf

def _is_class_modality(mod):
  # TODO(lukaszkaiser): should be based on type, like CLASS_LABEL, not string.
  prefix = "class_label_modality_"
  if len(mod.name) < len(prefix):
    return False
  return mod.name[:len(prefix)] == prefix


class AttentionLM():
  """Attention net.  See file docstring."""

  def __init__(self, hparams, mode, problem_hparams, problem_idx=0,
               data_parallelism=None, ps_devices=None):
    """Create a T2TModel.

    Args:
      hparams: a hyperparameters object.
      mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
      problem_hparams: a hyperparameters object.
      problem_idx: an integer.
      data_parallelism: a expert_utils.parallelism
        (specifies devices for data parallelism).
      ps_devices: a list of devices to be used for experts

    Returns:
      a T2TModel
    """
    if data_parallelism is None:
      data_parallelism = eu.Parallelism([""])
    if ps_devices is None:
      ps_devices = [""]
    hparams = copy.copy(hparams)
    hparams.add_hparam("mode", mode)
    # when not in training mode, set all forms of dropout to zero.
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
      for key in hparams.values():
        if key[-len("dropout"):] == "dropout":
          setattr(hparams, key, 0.0)
    self._hparams = hparams
    self._data_parallelism = data_parallelism
    self._num_datashards = data_parallelism.n
    self._ps_devices = ps_devices
    self._problem_hparams = problem_hparams
    self._problem_idx = problem_idx
    self._create_modalities(problem_hparams, hparams)

  def _create_modalities(self, problem_hparams, hparams):
    """Construct modalities in problem_hparams."""
    problem_hparams.input_modality = {}

    target_modality_spec = problem_hparams.target_modality
    if isinstance(target_modality_spec, modalities.SymbolModality):
      return
    target_modality = modalities.SymbolModality(hparams, target_modality_spec[1])
    problem_hparams.target_modality = target_modality

  @property
  def has_input(self):
    return self._problem_hparams.input_modality

  def infer(self, features=None, decode_length=50, beam_size=1,
            top_beams=1, last_position_only=False, alpha=0.0):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      last_position_only: a boolean, speed-up by computing last position only.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`.
    """
    if not self.has_input:
      # since there is no input, it is more interesting to see randomly
      # generated sequences, than to see the most likely sequence repeatedly.
      beam_size = 1
      self._hparams.sampling_method = "random"
    if _is_class_modality(
        self._hparams.problems[self._problem_idx].target_modality):
      beam_size = 1  # No use to run beam-search for a single class.
    if beam_size == 1:
      tf.logging.info("Greedy Decoding")
      return self._greedy_infer(features, decode_length, last_position_only)
    else:
      tf.logging.info("Beam Decoding with beam size %d" % beam_size)
      return self._beam_decode(features, decode_length, beam_size, top_beams,
                               last_position_only, alpha)

  def eval(self, features = None):

    start_time = time.time()
    dp = self._data_parallelism

    sharded_features = self._shard_features(features)

    # Construct the model bottom for inputs.
    transformed_features = {}
    all_previous_modalities = []

    # Target space id just gets copied to every shard.
    if "target_space_id" in features:
      transformed_features["target_space_id"] = [features["target_space_id"]] * self._num_datashards

    # Targets are transformed by the autoregressive part of the modality
    previous_tgt_modalities = [
        self._hparams.problems[i].target_modality.name
        for i in xrange(self._problem_idx)
    ]
    all_previous_modalities.extend(previous_tgt_modalities)

    target_modality = self._problem_hparams.target_modality
    target_reuse = target_modality.name in previous_tgt_modalities
    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      transformed_features["targets"] = target_modality.targets_bottom_sharded(sharded_features["targets"], dp)

    # Construct the model body.
    with tf.variable_scope("body", reuse=self._problem_idx > 0):
      body_outputs, extra_loss = self.model_fn_body_sharded(transformed_features)

    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      sharded_logits,\
      training_loss = target_modality.top_sharded(body_outputs, sharded_features["targets"], self._data_parallelism)

    tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
    return {
        "word_loss" : training_loss,
        "extra_loss" : extra_loss
    }


  def _beam_decode(self, features, decode_length, beam_size, top_beams,
                   last_position_only, alpha):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      last_position_only: a boolean, speed-up by computing last position only.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """

    def symbols_to_logits_fn(ids):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])

      features["targets"] = ids
      self._coverage = None
      sharded_logits, _, _ = self.model_fn(features, False, last_position_only=last_position_only)
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      logits = sharded_logits[0]  # Assuming we have one shard.
      if last_position_only:
        return tf.squeeze(logits, axis=[1, 2, 3])
      current_output_position = tf.shape(ids)[1] - 1  # -1 due to the pad above.
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    batch_size = tf.shape(features["inputs"])[0]
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    inputs_old = features["inputs"]
    features["inputs"] = tf.expand_dims(features["inputs"], 1)
    if len(features["inputs"].shape) < 5:
      features["inputs"] = tf.expand_dims(features["inputs"], 4)
    # Expand the inputs in to the beam size.
    features["inputs"] = tf.tile(features["inputs"], [1, beam_size, 1, 1, 1])
    s = tf.shape(features["inputs"])
    features["inputs"] = tf.reshape(features["inputs"], [s[0] * s[1], s[2], s[3], s[4]])

    target_modality = self._hparams.problems[self._problem_idx].target_modality
    vocab_size = target_modality.top_dimensionality
    # Setting decode length to input length + decode_length
    decode_length = tf.shape(features["inputs"])[1] + tf.constant(decode_length)
    ids, scores = beam_search.beam_search(symbols_to_logits_fn, initial_ids,
                                          beam_size, decode_length, vocab_size, alpha)

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator!
    features["inputs"] = inputs_old

    # Return `top_beams` decodings (also remove initial id from the beam search)
    return_scores = False  # TODO(lukaszkaiser): make it work multi-problem.
    if top_beams == 1:
      if return_scores:
        return {"outputs": ids[:, 0, 1:], "scores": scores}
      return ids[:, 0, 1:]
    else:
      if return_scores:
        return {"outputs": ids[:, :top_beams, 1:], "scores": scores}
      return ids[:, :top_beams, 1:]

  def _greedy_infer(self, features, decode_length, last_position_only):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      last_position_only: a boolean, speed-up by computing last position only.

    Returns:
       samples: an integer `Tensor`.
    """
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)
    if not self.has_input:
      features["partial_targets"] = tf.to_int64(features["inputs"])

    def infer_step(recent_output, _):
      """Inference step."""
      recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if last_position_only is set (dangerous).
      samples = self.sample(features, last_position_only=last_position_only)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      if last_position_only:
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:, tf.shape(recent_output)[1], :, :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)
      samples.set_shape([None, None, None, 1])
      return samples

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.convert_to_tensor(features["partial_targets"])
    else:
      batch_size = tf.shape(features["inputs"])[0]
      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0], tf.shape(initial_output))
    if _is_class_modality(self._hparams.problems[self._problem_idx].target_modality):
      decode_length = 1
    else:
      decode_length = tf.shape(features["inputs"])[1] + decode_length
    result = tf.foldl(
        infer_step,
        tf.range(decode_length),
        initializer=initial_output,
        back_prop=False,
        parallel_iterations=1)
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return result

  def sample(self, features, last_position_only=False):
    """Run the model and extract samples.

    Args:
      features: an map of string to `Tensor`.
      last_position_only: a boolean, speed-up by computing last position only.

    Returns:
       samples: an integer `Tensor`.
    """
    sharded_logits, _, _ = self.model_fn(features, False, last_position_only=last_position_only)
    if self._hparams.sampling_method == "argmax":
      sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
    else:
      assert self._hparams.sampling_method == "random"

      def _multinomial_squeeze(logits):
        reshaped_logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices, tf.shape(logits)[:logits.get_shape().ndims - 1])
        return choices

      sharded_samples = self._data_parallelism(_multinomial_squeeze,
                                               sharded_logits)
    return tf.concat(sharded_samples, 0)

  def _shard_features(self, features):  # pylint: disable=missing-docstring
    sharded_features = dict()
    for k, v in six.iteritems(features):
      v = tf.convert_to_tensor(v)
      if not v.shape.as_list():
        v = tf.expand_dims(v, axis=-1)
        v = tf.tile(v, [self._num_datashards])
      sharded_features[k] = self._data_parallelism(tf.identity, tf.split(v, self._num_datashards, 0))
    return sharded_features

  def model_fn(self, features, skip=False, last_position_only=False):

    start_time = time.time()
    dp = self._data_parallelism

    sharded_features = self._shard_features(features)

    # Construct the model bottom for inputs.
    transformed_features = {}
    all_previous_modalities = []

    for key, input_modality in six.iteritems(
        self._problem_hparams.input_modality):
      previous_modalities = [
          self._hparams.problems[i].input_modality[key].name
          for i in xrange(self._problem_idx)
      ]
      all_previous_modalities.extend(previous_modalities)
      do_reuse = input_modality.name in all_previous_modalities
      with tf.variable_scope(input_modality.name, reuse=do_reuse):
        transformed_features[key] = input_modality.bottom_sharded(
            sharded_features[key], dp)
      all_previous_modalities.append(input_modality.name)

    # Target space id just gets copied to every shard.
    if "target_space_id" in features:
      transformed_features["target_space_id"] = [features["target_space_id"]
                                                ] * self._num_datashards

    # Targets are transformed by the autoregressive part of the modality
    previous_tgt_modalities = [
        self._hparams.problems[i].target_modality.name
        for i in xrange(self._problem_idx)
    ]
    all_previous_modalities.extend(previous_tgt_modalities)

    target_modality = self._problem_hparams.target_modality
    target_reuse = target_modality.name in previous_tgt_modalities
    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      transformed_features["targets"] = target_modality.targets_bottom_sharded(
          sharded_features["targets"], dp)

    # Construct the model body.
    with tf.variable_scope("body", reuse=self._problem_idx > 0):
      if skip:
        body_outputs, extra_loss = transformed_features["targets"], 0.0
      else:
        body_outputs, extra_loss = self.model_fn_body_sharded(
            transformed_features)

    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      if not last_position_only:
        sharded_logits, training_loss = (target_modality.top_sharded(
            body_outputs, sharded_features["targets"], self._data_parallelism))

        training_loss *= self._problem_hparams.loss_multiplier
      else:
        # Take body outputs for the last position only, and targets too.
        # TODO(lukaszkaiser): warning, this doesn't work for all modalities!
        last_position_body_outputs = [
            tf.expand_dims(body_shard[:, -1, :, :], axis=[1])
            for body_shard in body_outputs
        ]
        last_position_targets = [
            tf.expand_dims(target_shard[:, -1:, :, :], axis=[1])
            for target_shard in sharded_features["targets"]
        ]
        sharded_logits, training_loss = (target_modality.top_sharded(
            last_position_body_outputs, last_position_targets,
            self._data_parallelism))

        training_loss = None

    tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
    return sharded_logits, training_loss, extra_loss

  def model_fn_body_sharded(self, sharded_features):

    with tf.name_scope("model"):
      datashard_to_features = [{k: v[d] for k, v in six.iteritems(sharded_features)} for d in xrange(self._num_datashards)]
      output = self._data_parallelism(self.model_fn_body, datashard_to_features)
      if isinstance(output, tuple):
        loss = tf.reduce_mean(output[1])
        output = output[0]
      else:
        loss = 0.0
      return output, loss

  def model_fn_body_enc(self, features):
      hparams = self._hparams
      targets = features["targets"]
      targets = tf.squeeze(targets, 2)

      (encoder_input, encoder_self_attention_bias, _) = attention_lm_prepare_encoder(targets, hparams)

      def residual_fn(x, y):
          return common_layers.layer_norm(x + tf.nn.dropout(y, 1.0 - hparams.residual_dropout))

      encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
      encoder_output = attention_lm_encoder(encoder_input, residual_fn, encoder_self_attention_bias, hparams)
      encoder_output = tf.expand_dims(encoder_output, 2)

      return encoder_output

  def model_fn_body(self, features):
    # Remove dropout if not training
    hparams = self._hparams
    targets = features["targets"]
    targets = tf.squeeze(targets, 2)

    (decoder_input, decoder_self_attention_bias) = attention_lm_prepare_decoder(targets, hparams)

    def residual_fn(x, y):
      return common_layers.layer_norm(x + tf.nn.dropout(y, 1.0 - hparams.residual_dropout))

    decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.residual_dropout)
    decoder_output = attention_lm_decoder(decoder_input, residual_fn, decoder_self_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output

  @property
  def hparams(self):
    return self._hparams


def attention_lm_prepare_encoder(targets, hparams):
    #ishape_static = targets.shape.as_list()
    encoder_padding = common_attention.embedding_to_padding(targets)
    encoder_self_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)
    encoder_input = common_layers.shift_left_3d(targets)

    #emb_target_space = common_layers.embedding(target_space, 32, ishape_static[-1], name = "target_space_embedding")
    #emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    #encoder_input += emb_target_space
    if hparams.pos == "timing":
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    return (encoder_input, encoder_self_attention_bias, encoder_padding)


def attention_lm_encoder(encoder_input, residual_fn, encoder_self_attention_bias, hparams, name = "encoder"):
    x = encoder_input
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                x = residual_fn(
                    x,
                    common_attention.multihead_attention(
                        x,
                        None,
                        encoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        summaries = summaries,
                        name = "encoder_self_attention"
                    )
                )
                x = residual_fn(
                    x,
                    common_layers.conv_hidden_relu(
                            x,
                            hparams.filter_size,
                            hparams.hidden_size,
                            dropout=hparams.relu_dropout
                    )
                )
    return x


def attention_lm_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a Tensor, containing large negative values
    to implement masked attention and possibly baises for diagonal alignments
  """
  decoder_self_attention_bias = (common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  decoder_input = common_layers.shift_left_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


def attention_lm_decoder(decoder_input, residual_fn, decoder_self_attention_bias, hparams, name = "decoder"):
  """A stack of attention_lm layers.

  Args:
    decoder_input: a Tensor
    residual_fn: a function from (layer_input, layer_output) -> combined_output
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  x = decoder_input
  # Summaries don't work in multi-problem setting yet.
  summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = residual_fn(
            x,
            common_attention.multihead_attention(
                x,
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                summaries=summaries,
                name="decoder_self_attention"))
        x = residual_fn(x,
                        common_layers.conv_hidden_relu(
                            x,
                            hparams.filter_size,
                            hparams.hidden_size,
                            dropout=hparams.relu_dropout))
  return x


def basic_params1():
  """A set of basic hyperparameters."""
  return tf.contrib.training.HParams(
      batch_size=4096,  # in tokens per batch per gpu
      # This flag controls the number of length buckets in the data reader.
      # Too many buckets slows down data reading - this needs fixing.
      # Too few buckets mean lots of wasted padding.
      # If this value is 1, we have buckets with maximum lengths:
      # [8, 12, 16, 24, 32, 48 ... (max_length or batch_size)]
      # If this value is 2, we have buckets with maximum lengths:
      # [8, 10, 12, 14, 16, 20, 24 ... (max_length or batch_size)]
      batching_mantissa_bits=1,
      num_hidden_layers=4,
      kernel_height=3,
      kernel_width=1,
      hidden_size=64,
      compress_steps=0,
      # All hyperparameters ending in "dropout" are automatically set to 0.0
      # when not in training mode.
      dropout=0.2,
      clip_grad_norm=2.0,
      initializer="orthogonal",
      initializer_gain=1.5,
      label_smoothing=0.1,
      optimizer="Adam",
      optimizer_adam_epsilon=1e-6,
      optimizer_adam_beta1=0.85,
      optimizer_adam_beta2=0.997,
      optimizer_momentum_momentum=0.9,
      weight_decay=0.1,
      weight_noise=0.0,
      learning_rate_decay_scheme="none",
      learning_rate_warmup_steps=100,
      learning_rate_cosine_cycle_steps=250000,
      learning_rate=0.1,
      sampling_method="argmax",  # "argmax" or "random"
      problem_choice="adaptive",  # "uniform", "adaptive", "distributed"
      multiply_embedding_mode="sqrt_depth",
      norm_type="none",  # "batch", layer", "noam", "none".
      layer_norm_epsilon=1e-6,
      symbol_modality_num_shards=16,
      # setting the max length in a minibatch. 0 means default behavior,
      # max_length = hparams.batch_size * length_multiplier
      max_length=0,
      # in SymbolModality, share the output embeddings and the softmax
      # variables.
      # You can also share the input embeddings with the output embeddings
      # by using a problem_hparams that uses the same modality object for
      # the input_modality and target_modality.
      shared_embedding_and_softmax_weights=int(False),
      # For each feature for which you want to override the default input
      # modality, add an entry to this semicolon-separated string. Entries are
      # formatted "feature_name:modality_type:modality_name", e.g.
      # "inputs:image:small_image_modality;other_inputs:audio:identity".
      input_modalities="",
      # To override the default target modality, specify
      # "modality_type:modality_name", e.g. "image:small_image_modality".
      target_modality="")


def attention_lm_base():
  """Set of hyperparameters."""
  hparams = basic_params1()
  hparams.hidden_size = 1024
  hparams.batch_size = 8192
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 2000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.0
  hparams.shared_embedding_and_softmax_weights = int(False)

  hparams.add_hparam("filter_size", 4096)  # Add new ones like this.
  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("residual_dropout", 0.1)
  hparams.add_hparam("pos", "timing")  # timing, none
  return hparams


def attention_lm_small():
  """Cheap model.

  on lm1b_32k:
     45M params
     2 steps/sec on  [GeForce GTX TITAN X]

  Returns:
    an hparams object.
  """
  hparams = attention_lm_base()
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.residual_dropout = 0.5
  return hparams
