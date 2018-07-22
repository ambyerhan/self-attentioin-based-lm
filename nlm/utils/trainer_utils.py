# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for trainer binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator

# Dependency imports

import numpy as np
import six
# pylint: disable=redefined-builtin
from six.moves import input
from six.moves import xrange
# pylint: enable=redefined-builtin

from nlm.data_generators import text_encoder
from nlm.utils import data_reader
from nlm.utils import expert_utils as eu
from nlm.utils import metrics

from nlm.models import attention_lm
from nlm.data_generators import problem_hparams

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.ops import init_ops
from tensorflow.python.training import saver
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib import framework as contrib_framework
from tensorflow.python.training import monitored_session
from tensorflow.python.framework import random_seed


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("registry_help", False, "If True, logs the contents of the registry and exits.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("model", "", "Which model to use.")
flags.DEFINE_string("hparams_set", "", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string("hparams", "", """A comma-separated list of `name=value` hyperparameter values. This flag
    is used to override hyperparameter settings either when manually selecting
    hyperparameters or when using Vizier. If a hyperparameter setting is
    specified by this flag then it must be a valid hyperparameter name for the
    model.""")
flags.DEFINE_string("problems", "", "Dash separated list of problems to solve.")
flags.DEFINE_string("data_dir", "/tmp/data", "Directory with training data.")
flags.DEFINE_integer("train_steps", 250000, "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_bool("eval_print", False, "Print eval logits and predictions.")
flags.DEFINE_integer("keep_checkpoint_max", 20, "How many recent checkpoints to keep.")
flags.DEFINE_bool("experimental_optimize_placement", False, "Optimize ops placement with experimental session options.")

# Distributed training flags
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "local_run", "Method of tf.contrib.learn.Experiment to run.")
flags.DEFINE_integer("local_eval_frequency", 2000, "Run evaluation every this steps during local training.")
flags.DEFINE_bool("locally_shard_to_cpu", False, "Use CPU as a sharding device runnning locally. This allows "
                  "to test sharded model construction on a machine with 1 GPU.")
flags.DEFINE_bool("daisy_chain_variables", True, "copy variables around in a daisy chain")
flags.DEFINE_bool("sync", False, "Sync compute on PS.")
flags.DEFINE_string("worker_job", "/job:worker", "name of worker job")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_float("worker_gpu_memory_fraction", 1., "Fraction of GPU memory to allocate.")
flags.DEFINE_integer("ps_gpu", 0, "How many GPUs to use per ps.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining gpus. e.g. \"1 3 2 4\"")
flags.DEFINE_string("ps_job", "/job:ps", "name of ps job")
flags.DEFINE_integer("ps_replicas", 0, "How many ps replicas.")

# Decode flags
flags.DEFINE_bool("decode_use_last_position_only", False, "In inference, use last position only for speedup.")
flags.DEFINE_bool("decode_interactive", False, "Interactive local inference mode.")
flags.DEFINE_bool("decode_save_images", False, "Save inference input images.")
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None, "Path to inference output file")
flags.DEFINE_integer("decode_shards", 1, "How many shards to decode.")
flags.DEFINE_integer("decode_problem_id", 0, "Which problem to decode.")
flags.DEFINE_integer("decode_extra_length", 50, "Added decode length.")
flags.DEFINE_integer("decode_batch_size", 32, "Batch size for decoding. "
                     "The decodes will be written to <filename>.decodes in"
                     "format result\tinput")
flags.DEFINE_integer("decode_beam_size", 4, "The beam size for beam decoding")
flags.DEFINE_float("decode_alpha", 0.6, "Alpha for length penalty")
flags.DEFINE_bool("decode_return_beams", False, "Whether to return 1 (False) or all (True) beams. The \n "
                  "output file will have the format <beam1>\t<beam2>..\t<input>")


def _save_until_eos(hyp):
  """Strips everything after the first <EOS> token, which is normally 1."""
  try:
    index = list(hyp).index(text_encoder.EOS_TOKEN)
    return hyp[0:index]
  except ValueError:
    # No EOS_TOKEN: return the array as-is.
    return hyp


def create_experiment(output_dir, data_dir, model_name, train_steps, eval_steps):
  hparams = create_hparams(FLAGS.hparams_set, data_dir)
  estimator, input_fns = create_experiment_components(
      hparams=hparams,
      output_dir=output_dir,
      data_dir=data_dir,
      model_name=model_name)
  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=input_fns["train"],
      eval_input_fn=input_fns["eval"],
      eval_metrics=metrics.create_evaluation_metrics(FLAGS.problems.split("-")),
      train_steps=train_steps,
      eval_steps=eval_steps,
      min_eval_frequency=FLAGS.local_eval_frequency,
      train_monitors=[])


def create_experiment_components(hparams, output_dir, data_dir, model_name):
  """Constructs and returns Estimator and train/eval input functions."""
  tf.logging.info("Creating experiment, storing model files in %s", output_dir)

  num_datashards = data_parallelism().n
  train_input_fn = get_input_fn(
      mode=tf.contrib.learn.ModeKeys.TRAIN,
      hparams=hparams,
      data_file_patterns=get_datasets_for_mode(data_dir, tf.contrib.learn.ModeKeys.TRAIN),
      num_datashards=num_datashards)

  eval_input_fn = get_input_fn(
      mode=tf.contrib.learn.ModeKeys.EVAL,
      hparams=hparams,
      data_file_patterns=get_datasets_for_mode(data_dir, tf.contrib.learn.ModeKeys.EVAL),
      num_datashards=num_datashards)
  estimator = tf.contrib.learn.Estimator(
      model_fn=model_builder(model_name, hparams=hparams),
      model_dir=output_dir,
      config=tf.contrib.learn.RunConfig(
          master=FLAGS.master,
          model_dir=output_dir,
          gpu_memory_fraction=FLAGS.worker_gpu_memory_fraction,
          session_config=session_config(),
          keep_checkpoint_max=FLAGS.keep_checkpoint_max))
  # Store the hparams in the estimator as well
  estimator.hparams = hparams
  return estimator, {"train": train_input_fn, "eval": eval_input_fn}


def create_hparams(hparams_set, data_dir):
  hparams = None
  assert hparams_set in ["lm_base", "lm_small"]
  if hparams_set == "lm_base":
      hparams = attention_lm.attention_lm_base()
  elif hparams_set == "lm_small":
      hparams = attention_lm.attention_lm_small()
  else:
      tf.logging.info("haprams set is error!")
      exit()
  hparams.add_hparam("data_dir", data_dir)
  if FLAGS.hparams:
    hparams = hparams.parse(FLAGS.hparams)

  # Add hparams for the problems
  hparams.problems = []
  p_hparams = problem_hparams.lmptb_10k(hparams)

  hparams.problems.append(p_hparams)
  return hparams


def run(data_dir, model, output_dir, train_steps, eval_steps, schedule):
  exp = create_experiment(
      output_dir=output_dir,
      data_dir=data_dir,
      model_name=model,
      train_steps=train_steps,
      eval_steps=eval_steps)

  assert schedule == "local_run"
  run_locally(exp)


def validate_flags():
  if not (FLAGS.hparams_set or FLAGS.hparams_range):
    raise ValueError("Must specify either --hparams_set or --hparams_range.")
  if not FLAGS.schedule:
    raise ValueError("Must specify --schedule.")
  if not FLAGS.output_dir:
    FLAGS.output_dir = "/tmp/tensor2tensor"
    tf.logging.warning("It is strongly recommended to specify --output_dir. "
                       "Using default output_dir=%s.", FLAGS.output_dir)


def session_config():
  """The TensorFlow Session config to use."""
  graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
      opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))

  if FLAGS.experimental_optimize_placement:
    rewrite_options = tf.RewriterConfig(optimize_tensor_layout=True)
    rewrite_options.optimizers.append("pruning")
    rewrite_options.optimizers.append("constfold")
    rewrite_options.optimizers.append("layout")
    graph_options = tf.GraphOptions(
        rewrite_options=rewrite_options, infer_shapes=True)

  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=FLAGS.worker_gpu_memory_fraction)

  config = tf.ConfigProto(allow_soft_placement=True,
                          graph_options=graph_options,
                          gpu_options=gpu_options)
  return config


def model_builder(model, hparams):

  def initializer():
    if hparams.initializer == "orthogonal":
      return tf.orthogonal_initializer(gain=hparams.initializer_gain)
    elif hparams.initializer == "uniform":
      max_val = 0.1 * hparams.initializer_gain
      return tf.random_uniform_initializer(-max_val, max_val)
    elif hparams.initializer == "normal_unit_scaling":
      return init_ops.variance_scaling_initializer(
          hparams.initializer_gain, mode="fan_avg", distribution="normal")
    elif hparams.initializer == "uniform_unit_scaling":
      return init_ops.variance_scaling_initializer(
          hparams.initializer_gain, mode="fan_avg", distribution="uniform")
    else:
      raise ValueError("Unrecognized initializer: %s" % hparams.initializer)

  def learning_rate_decay():
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(
        hparams.learning_rate_warmup_steps * FLAGS.worker_replicas)
    step = tf.to_float(tf.contrib.framework.get_global_step())
    if hparams.learning_rate_decay_scheme == "noam":
      return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
          (step + 1) * warmup_steps**-1.5, (step + 1)**-0.5)
    elif hparams.learning_rate_decay_scheme == "exp100k":
      return 0.94**(step // 100000)
    elif hparams.learning_rate_decay_scheme == "cosine":
      cycle_steps = hparams.learning_rate_cosine_cycle_steps
      return 0.5 * (1 + tf.cos(np.pi * (step % cycle_steps) / cycle_steps))

    inv_base = tf.exp(tf.log(0.01) / warmup_steps)
    inv_decay = inv_base**(warmup_steps - step)
    if hparams.learning_rate_decay_scheme == "sqrt":
      decay = _sqrt_decay(step - warmup_steps)
    elif hparams.learning_rate_decay_scheme == "exp10k":
      decay = _exp_decay_after(step - warmup_steps, 0.9995,
                               FLAGS.train_steps - warmup_steps - 10000)
    elif hparams.learning_rate_decay_scheme == "exp50k":
      decay = _exp_decay_after(step - warmup_steps, 0.99995,
                               FLAGS.train_steps - warmup_steps - 50000)
    elif hparams.learning_rate_decay_scheme == "exp500k":
      decay = _exp_decay_after(step - warmup_steps, 0.9999955,
                               FLAGS.train_steps - warmup_steps - 500000)
    elif hparams.learning_rate_decay_scheme == "none":
      decay = tf.constant(1.0)
    else:
      raise ValueError("Unrecognized learning rate decay scheme: %s" %
                       hparams.learning_rate_decay_scheme)
    return tf.cond(
        step < warmup_steps,
        lambda: inv_decay,
        lambda: decay,
        name="learning_rate_decay_warump_cond")

  def model_fn(features, targets, mode):

    if mode == tf.contrib.learn.ModeKeys.INFER:
      if FLAGS.decode_from_file:
        features = _decode_input_tensor_to_features_dict(features, hparams) # AMBYER >> set the hparams to the features, which is a dict

    run_info = dict()
    run_info["problem_choice"] = features["problem_choice"]

    if targets is not None:
      features["targets"] = targets

    dp = data_parallelism()

    tf.get_variable_scope().set_initializer(initializer())

    n = 0
    model_class = attention_lm.AttentionLM(hparams, mode, hparams.problems[n],
                                           n, dp, _ps_devices(all_workers=True))

    """Build the model for the n-th problem, plus some added variables."""
    if mode == tf.contrib.learn.ModeKeys.INFER:
      result_list = model_class.eval(features)
      training_loss, extra_loss = result_list["word_loss"], result_list["extra_loss"]
      total_loss = training_loss + extra_loss
      total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")

      ret = {"word_loss": total_loss}, None, None
      return ret

    sharded_logits, training_loss, extra_loss = model_class.model_fn(features, skip=False)
    with tf.variable_scope("losses_avg", reuse=True):
      loss_moving_avg = tf.get_variable("problem_%d/training_loss" % n)
      o1 = loss_moving_avg.assign(loss_moving_avg * 0.9 + training_loss * 0.1)
      loss_moving_avg = tf.get_variable("problem_%d/extra_loss" % n)
      o2 = loss_moving_avg.assign(loss_moving_avg * 0.9 + extra_loss * 0.1)
      loss_moving_avg = tf.get_variable("problem_%d/total_loss" % n)
      total_loss = training_loss + extra_loss
      o3 = loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1)
    with tf.variable_scope("train_stats"):  # Count steps for this problem.
      problem_steps = tf.get_variable("problem_%d_steps" % n, initializer=0, trainable=False)
      o4 = problem_steps.assign_add(1)
    with tf.control_dependencies([o1, o2, o3, o4]):  # Make sure the ops run.
      # Ensure the loss is a scalar here.
      total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")
    result_list = [total_loss] + sharded_logits  # Need to flatten for cond later.

    sharded_logits, total_loss = result_list[1:], result_list[0]
    if mode == tf.contrib.learn.ModeKeys.EVAL:
      logits = tf.concat(sharded_logits, 0)
      if FLAGS.eval_print:
        logits = tf.Print(logits, [features["inputs"], logits], "EVAL PRINT", summarize=10000)
      # For evaluation, return the logits layer as our predictions.
      run_info["predictions"] = logits
      train_op = None
      return run_info, total_loss, None

    assert mode == tf.contrib.learn.ModeKeys.TRAIN

    # Some training statistics.
    with tf.name_scope("training_stats"):
      learning_rate = hparams.learning_rate * learning_rate_decay()
      learning_rate /= math.sqrt(float(FLAGS.worker_replicas))
      tf.summary.scalar("learning_rate", learning_rate)
      global_step = tf.to_float(tf.contrib.framework.get_global_step())
      for n in xrange(len(hparams.problems)):
        with tf.variable_scope("losses_avg", reuse=True):
          total_loss_var = tf.get_variable("problem_%d/total_loss" % n)
          training_loss_var = tf.get_variable("problem_%d/training_loss" % n)
          extra_loss_var = tf.get_variable("problem_%d/extra_loss" % n)
        tf.summary.scalar("loss_avg_%d/total_loss" % n, total_loss_var)
        tf.summary.scalar("loss_avg_%d/training_loss" % n, training_loss_var)
        tf.summary.scalar("loss_avg_%d/extra_loss" % n, extra_loss_var)
        with tf.variable_scope("train_stats", reuse=True):
          nth_steps = tf.get_variable("problem_%d_steps" % n, dtype=tf.int32)
        tf.summary.scalar("problem_%d_frequency" % n, tf.to_float(nth_steps) / (global_step + 1.0))

    # Log trainable weights and add decay.
    total_size, weight_decay_loss = 0, 0.0
    all_weights = {v.name: v for v in tf.trainable_variables()}
    for v_name in sorted(list(all_weights)):
      v = all_weights[v_name]
      v_size = int(np.prod(np.array(v.shape.as_list())))
      tf.logging.info("Weight    %s\tshape    %s\tsize    %d", v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
      total_size += v_size
      if hparams.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
        # Add weight regularization if set and the weight is not a bias (dim>1).
        with tf.device(v._ref().device):  # pylint: disable=protected-access
          v_loss = tf.nn.l2_loss(v) / v_size
        weight_decay_loss += v_loss
      is_body = len(v_name) > 5 and v_name[:5] == "body/"
      if hparams.weight_noise > 0.0 and is_body:
        # Add weight noise if set in hparams.
        with tf.device(v._ref().device):  # pylint: disable=protected-access
          scale = learning_rate * 0.001
          noise = tf.truncated_normal(v.shape) * hparams.weight_noise * scale
          noise_op = v.assign_add(noise)
        with tf.control_dependencies([noise_op]):
          total_loss = tf.identity(total_loss)
    tf.logging.info("Total trainable variables size: %d", total_size)
    if hparams.weight_decay > 0.0:
      total_loss += weight_decay_loss * hparams.weight_decay
    total_loss = tf.identity(total_loss, name="total_loss")

    # Define the train_op for the TRAIN mode.
    opt = _ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
    tf.logging.info("Computing gradients for global model_fn.")
    train_op = tf.contrib.layers.optimize_loss(
        name="training",
        loss=total_loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=learning_rate,
        clip_gradients=hparams.clip_grad_norm or None,
        optimizer=opt,
        colocate_gradients_with_ops=True)

    tf.logging.info("Global model_fn finished.")
    return run_info, total_loss, train_op

  return model_fn


def run_locally(exp):
  if exp.train_steps > 0:
    # Train
    tf.logging.info("Performing local training.")
    exp.train_and_evaluate()
  else: # Predict FLAGS.decode_from_file is not None:
    estimator = exp.estimator
    hparams = estimator.hparams
    cfg = estimator.config
    decode_from_file(hparams, FLAGS.decode_from_file, model_builder('attention_lm', hparams), cfg, FLAGS.output_dir)


def decode_from_file(hparams, filename, model_fn, cfg, model_dir):
  """Compute predictions on entries in filename and write them out."""
  problem_id = FLAGS.decode_problem_id
  targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = _get_sorted_inputs(filename)
  num_decode_batches = (len(sorted_inputs) - 1) // FLAGS.decode_batch_size + 1 # AMBYER >> (len(inputs) - 1 // batch_size) + 1
  input_fn = _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs, targets_vocab)

  decodes = []
  for _ in range(num_decode_batches):
    result_iter = _decode_infer_model(input_fn = input_fn.__next__, as_iterable = True,
                                      model_fn = model_fn, runCfg = cfg, model_dir = model_dir)
    for result in result_iter:
      decodes.append(result["word_loss"])
  for i, loss in enumerate(decodes):
      tf.logging.info("[%d/%d] loss = %.3f, ppl = %.3f" % (i, len(decodes), loss, np.exp(loss)))
  ave_loss = sum(decodes) / len(decodes)
  tf.logging.info("the loss/ppl on set %s = %.3f/%.3f" % (filename, ave_loss, np.exp(ave_loss)))



def _decode_infer_model(input_fn, feed_fn=None, model_fn = None, runCfg = None,
                        model_dir= None, outputs=None, as_iterable=True, iterate_batches=False):
  # Check that model has been trained.
  checkpoint_path = saver.latest_checkpoint(model_dir)
  if not checkpoint_path:
      raise NotFittedError("Couldn't find trained model at %s." % model_dir)
  with tf.Graph().as_default() as g:
    random_seed.set_random_seed(runCfg.tf_random_seed)
    contrib_framework.create_global_step(g)
    features = input_fn()
    model_result, _, _ = model_fn(features, None, mode = tf.contrib.learn.ModeKeys.INFER)
    mon_sess = monitored_session.MonitoredSession(
        session_creator=monitored_session.ChiefSessionCreator(
            checkpoint_filename_with_path=checkpoint_path,
            scaffold=None, config=runCfg._session_config))

    return _decode_predict_generator(mon_sess, model_result, feed_fn, iterate_batches)

def _decode_predict_generator(mon_sess, predictions, feed_fn, iterate_batches):
    with mon_sess:
      while not mon_sess.should_stop():
        preds = mon_sess.run(predictions, feed_fn() if feed_fn else None)

        yield preds
        if _decode_is_input_constant(feed_fn, mon_sess.graph):
          return

def _decode_is_input_constant(feed_fn, graph):
    if graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      return False
    if feed_fn is not None:
      return False
    return True


def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs, vocabulary):
  tf.logging.info(" batch %d" % num_decode_batches)
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    #batch_labels = []
    for inputs in sorted_inputs[b * FLAGS.decode_batch_size: (b + 1) * FLAGS.decode_batch_size]:
      words = inputs.strip().split(' ')
      input_ids = [vocabulary._token_to_id.get(wd, text_encoder.UNK_TOKEN) for wd in words]
      input_ids.append(text_encoder.EOS_TOKEN)

      batch_inputs.append(input_ids)
      #batch_labels.append(input_ids[1:])
      if len(input_ids) > batch_length: # AMBYER>>let the max length of the sentence be the batch_length
        batch_length = len(input_ids)
    final_batch_inputs = []
    #final_batch_labels = []
    #for input_ids, label_ids in zip(batch_inputs, batch_labels):
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      #assert len(label_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids)) # AMBYER >> padding
      #y = label_ids + [0] * (batch_length - len(label_ids))
      final_batch_inputs.append(x)
      #final_batch_labels.append(y)

    yield {
        "inputs": np.array(final_batch_inputs),
        "problem_choice": np.array(problem_id)
    }


def _get_sorted_inputs(filename):
  """Returning inputs sorted according to length.

  Args:
    filename: path to file with inputs, 1 per line.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")
  # read file and sort inputs according them according to input length.
  if FLAGS.decode_shards > 1:
    decode_filename = filename + ("%.2d" % FLAGS.worker_id)
  else:
    decode_filename = filename
  inputs = [line.strip() for line in tf.gfile.Open(decode_filename)]
  input_lens = [(i, len(line.strip().split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1)) # AMBEYR>>sorted by the sentlen
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i # AMBYER>>original_pos : sorted_pos
  return sorted_inputs, sorted_keys


def get_datasets_for_mode(data_dir, mode):
  return data_reader.get_datasets("lmptb_10k", data_dir, mode)


def _decode_input_tensor_to_features_dict(feature_map, hparams):
  #assert tf.contrib.learn.ModeKeys.INFER
  #tf.logging.info("\t\t[Ambyer] >>> decoding input feature to tensors")
  inputs = tf.constant(feature_map["inputs"])

  def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
    p_hparams = hparams.problems[problem_choice]
    # Add a third empty dimension dimension
    x = tf.expand_dims(tf.expand_dims(x, axis=[2]), axis = [3])
    x = tf.to_int32(x)

    return (tf.constant(p_hparams.target_space_id), x)

  target_space_id, x = input_fn(feature_map["problem_choice"])

  features = {}
  features["problem_choice"] = feature_map["problem_choice"]
  features["target_space_id"] = target_space_id
  features["decode_length"] = (tf.shape(x)[1] + 50)
  features["targets"] = x
  return features


def get_input_fn(mode, hparams, data_file_patterns=None,
                 num_datashards=None, fixed_problem=None):

  def input_fn():
    """Supplies input to our model. """
    problem_count, batches = len(data_file_patterns), []
    with tf.name_scope("input_queues"):
      for n in xrange(problem_count):
        if fixed_problem is not None and n != fixed_problem:
          continue
        with tf.name_scope("problem_%d" % n):
          with tf.device("/cpu:0"):  # Input queues are on CPU.
            capacity = hparams.problems[n].max_expected_batch_size_per_shard
            capacity *= num_datashards
            examples = data_reader.input_pipeline(data_file_patterns[n], capacity, mode)
            drop_long_sequences = mode == tf.contrib.learn.ModeKeys.TRAIN
            batch_size_multiplier = hparams.problems[n].batch_size_multiplier
            feature_map = data_reader.batch_examples(
                examples,
                data_reader.hparams_to_batching_scheme(
                    hparams,
                    shard_multiplier=num_datashards,
                    drop_long_sequences=drop_long_sequences,
                    length_multiplier=batch_size_multiplier))

        # Reverse inputs and targets features if the problem was reversed.
        if hparams.problems[n].was_reversed:
          inputs = feature_map["inputs"]
          targets = feature_map["targets"]
          feature_map["inputs"] = targets
          feature_map["targets"] = inputs

        # Use the inputs as the targets if the problem is a copy problem.
        if hparams.problems[n].was_copy:
          feature_map["targets"] = feature_map["inputs"]

        # Ensure inputs and targets are proper rank.
        while len(feature_map["inputs"].get_shape()) != 4:
          feature_map["inputs"] = tf.expand_dims(feature_map["inputs"], axis=-1)
        while len(feature_map["targets"].get_shape()) != 4:
          feature_map["targets"] = tf.expand_dims(
              feature_map["targets"], axis=-1)

        batches.append(
            (feature_map["inputs"], feature_map["targets"], tf.constant(n),
             tf.constant(hparams.problems[n].input_space_id),
             tf.constant(hparams.problems[n].target_space_id)))

    # We choose which problem to process.
    loss_moving_avgs = []  # Need loss moving averages for that.
    for n in xrange(problem_count):
      with tf.variable_scope("losses_avg"):
        loss_moving_avgs.append(
            tf.get_variable("problem_%d/total_loss" % n, initializer=100.0, trainable=False))
        tf.get_variable(
            "problem_%d/training_loss" % n, initializer=100.0, trainable=False)
        tf.get_variable(
            "problem_%d/extra_loss" % n, initializer=100.0, trainable=False)
    if fixed_problem is None:
      if (hparams.problem_choice == "uniform" or
          mode != tf.contrib.learn.ModeKeys.TRAIN):
        problem_choice = tf.random_uniform([], maxval=problem_count, dtype=tf.int32)
      elif hparams.problem_choice == "adaptive":
        loss_moving_avgs = tf.stack(loss_moving_avgs)
        problem_choice = tf.multinomial(tf.reshape(loss_moving_avgs, [1, -1]), 1)
        problem_choice = tf.to_int32(tf.squeeze(problem_choice))
      elif hparams.problem_choice == "distributed":
        assert FLAGS.worker_replicas >= problem_count
        assert FLAGS.worker_replicas % problem_count == 0
        problem_choice = tf.to_int32(FLAGS.worker_id % problem_count)
      else:
        raise ValueError("Value of hparams.problem_choice is %s and must be "
                         "one of [uniform, adaptive, distributed]" %
                         hparams.problem_choice)

      # Inputs and targets conditional on problem_choice.
      rand_inputs, rand_target, choice, inp_id, tgt_id = batches[0]
    else:
      problem_choice = tf.constant(fixed_problem)
      # Take the only constructed batch, which is the fixed_problem.
      rand_inputs, rand_target, choice, inp_id, tgt_id = batches[0]

    # Set shapes so the ranks are clear.
    rand_inputs.set_shape([None, None, None, None])
    rand_target.set_shape([None, None, None, None])
    choice.set_shape([])
    inp_id.set_shape([])
    tgt_id.set_shape([])
    #  Forced shape obfuscation is necessary for inference.
    if mode == tf.contrib.learn.ModeKeys.INFER:
      rand_inputs._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access
      rand_target._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access

    # Final feature map.
    rand_feature_map = {
        "inputs": rand_inputs,
        "problem_choice": choice,
        "input_space_id": inp_id,
        "target_space_id": tgt_id
    }
    if mode == tf.contrib.learn.ModeKeys.INFER:
      rand_feature_map["infer_targets"] = rand_target
      rand_target = None
    return rand_feature_map, rand_target

  return input_fn


class _ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams):
    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr, momentum=hparams.optimizer_momentum_momentum)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
    return self._opt.compute_gradients(
        loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

  def apply_gradients(self, gradients, global_step=None, name=None):
    return self._opt.apply_gradients(
        gradients, global_step=global_step, name=name)


def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")


def _ps_replicas(all_workers=False):
  if all_workers:
    return list(range(FLAGS.ps_replicas))
  # Worker K will be using replicas {0,...n-1} + K*n if we have n replicas.
  num_replicas = FLAGS.ps_replicas // FLAGS.worker_replicas
  return [d + FLAGS.worker_id * num_replicas for d in xrange(num_replicas)]


def _gpu_order(num_gpus):
  if FLAGS.gpu_order:
    ret = [int(s) for s in FLAGS.gpu_order.split(" ")]
    if len(ret) == num_gpus:
      return ret
  return list(range(num_gpus))


def _ps_gpus(all_workers=False):
  ps_gpus = []
  for d in _ps_replicas(all_workers=all_workers):
    ps_gpus.extend([(d, gpu) for gpu in _gpu_order(FLAGS.ps_gpu)])
  return ps_gpus


def _ps_devices(all_workers=False):
  """List of ps devices (where to put the experts).

  Args:
    all_workers: whether the list is for all async workers or just this one.

  Returns:
    a list of device names
  """
  if FLAGS.ps_replicas > 0:
    if FLAGS.ps_gpu > 0:
      return [
          FLAGS.ps_job + "/task:%d/GPU:%d" % (d, gpu)
          for (d, gpu) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      return [
          FLAGS.ps_job + "/task:%d" % d
          for d in _ps_replicas(all_workers=all_workers)
      ]
  else:
    if FLAGS.worker_gpu > 0:
      return ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
    else:
      return [""]


def data_parallelism(all_workers=False): # todo

  def _replica_device_setter(worker_device):
    if FLAGS.ps_replicas == 0:
      return worker_device
    return tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=FLAGS.ps_replicas,
        ps_device=FLAGS.ps_job + "/GPU:0" if FLAGS.ps_gpu > 0 else FLAGS.ps_job)

  if FLAGS.schedule == "local_run":
    assert not FLAGS.sync
    datashard_devices = ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
    if FLAGS.locally_shard_to_cpu:
      datashard_devices += ["cpu:0"]
    caching_devices = None
  elif FLAGS.sync:
    assert FLAGS.ps_replicas > 0
    datashard_devices = [
        _replica_device_setter(d) for d in _ps_devices(all_workers=all_workers)
    ]
    if FLAGS.ps_gpu > 0 and FLAGS.ps_replicas > 1:
      caching_devices = [
          FLAGS.ps_job + "/task:%d/cpu:0" % d
          for (d, _) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      caching_devices = None
  else:
    # old fashioned async - compute on worker
    if FLAGS.worker_gpu > 1:
      datashard_devices = [
          _replica_device_setter(FLAGS.worker_job + "/GPU:%d" % d)
          for d in _gpu_order(FLAGS.worker_gpu)
      ]
      caching_devices = [FLAGS.worker_job + "/GPU:0"] * FLAGS.worker_gpu
    else:
      datashard_devices = [_replica_device_setter(FLAGS.worker_job)]
      caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  return eu.Parallelism(
      datashard_devices,
      reuse=True,
      caching_devices=caching_devices,
      daisy_chain_variables=FLAGS.daisy_chain_variables)
