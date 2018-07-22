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

"""Data reader module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

# Dependency imports

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from nlm.data_generators import problem_hparams
from nlm.models import common_layers

import tensorflow as tf


def examples_queue(data_sources,
                   data_fields_to_features,
                   training,
                   capacity=32,
                   data_items_to_decoders=None,
                   data_items_to_decode=None):
  with tf.name_scope("examples_queue"):
    # Read serialized examples using slim parallel_reader.
    num_epochs = None if training else 1
    data_files = tf.contrib.slim.parallel_reader.get_data_files(data_sources)
    num_readers = min(4 if training else 1, len(data_files))
    _, example_serialized = tf.contrib.slim.parallel_reader.parallel_read(
        data_sources,
        tf.TFRecordReader,
        num_epochs=num_epochs,
        shuffle=training,
        capacity=2 * capacity,
        min_after_dequeue=capacity,
        num_readers=num_readers)

    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields_to_features
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields_to_features, data_items_to_decoders)

    if data_items_to_decode is None:
      data_items_to_decode = list(data_items_to_decoders)

    decoded = decoder.decode(example_serialized, items=data_items_to_decode)
    return {
        field: tensor
        for (field, tensor) in zip(data_items_to_decode, decoded)
    }


def input_pipeline(data_file_pattern, capacity, mode):
  """Input pipeline, returns a dictionary of tensors from queues."""
  # Read from image TFRecords if the file has "image" in its name.

  data_fields = {
      "inputs": tf.VarLenFeature(tf.int64),
      "targets": tf.VarLenFeature(tf.int64)
  }
  data_items_to_decoders = None

  # Create placeholders for input, rather than reading data from disk.
  if data_file_pattern is None:
    feature_map = {}
    for (field, tp) in data_fields:
      if field != "targets":
        feature_map[field] = tf.placeholder(
            dtype=tp, shape=[None] * 4, name=field)
    return feature_map

  # Now the non-trivial case construction.
  examples = examples_queue(
      [data_file_pattern],
      data_fields,
      training=(mode == tf.contrib.learn.ModeKeys.TRAIN),
      capacity=capacity,
      data_items_to_decoders=data_items_to_decoders)

  # We do not want int64s as they do are not supported on GPUs.
  return {k: tf.to_int32(v) for (k, v) in six.iteritems(examples)}


def batch_examples(examples, batching_scheme):
  """Given a queue of examples, create batches of examples with similar lengths.

  We assume that examples is a dictionary with string keys and tensor values,
  possibly coming from a queue, e.g., constructed by examples_queue above.
  Each tensor in examples is assumed to be 1D. We will put tensors of similar
  length into batches togeter. We return a dictionary with the same keys as
  examples, and with values being batches of size batch_size. If elements have
  different lengths, they are padded with 0s. This function is based on
  tf.contrib.training.bucket_by_sequence_length so see there for details.

  For example, if examples is a queue containing [1, 2, 3] and [4], then
  this function with batch_size=2 will return a batch [[1, 2, 3], [4, 0, 0]].

  Args:
    examples: a dictionary with string keys and 1D tensor values.
    batching_scheme: a dictionary containing
      "boundaries": a list of integers for the boundaries that will be
        used for bucketing; see tf.contrib.training.bucket_by_sequence_length
        for more details.
      "batch_sizes": a list of batch sizes corresponding to the buckets
      "max_length": an integer.  We drop sequences which are longer.

  Returns:
    A dictionary with the same keys as examples and with values being batches
    of examples padded with 0s, i.e., [batch_size x length] tensors.
  """
  with tf.name_scope("batch_examples"):
    # The queue to bucket on will be chosen based on maximum length.
    max_length = 0
    for v in examples.values():
      # For images the sequence length is the size of the spatial dimensions.
      sequence_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                         tf.shape(v)[0] * tf.shape(v)[1])
      max_length = tf.maximum(max_length, sequence_length)
    (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
        max_length,
        examples,
        batching_scheme["batch_sizes"],
        [b + 1 for b in batching_scheme["boundaries"]],
        capacity=2,  # Number of full batches to store, we don't need many.
        bucket_capacities=[2 * b for b in batching_scheme["batch_sizes"]],
        dynamic_pad=True,
        keep_input=(max_length <= batching_scheme["max_length"]))
    return outputs


def bucket_boundaries(max_length, min_length=8, mantissa_bits=2):
  """A default set of length-bucket boundaries."""
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x += 2**max(0, int(math.log(x, 2)) - mantissa_bits)
  return boundaries


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
  """A batching scheme based on model hyperparameters.

  Every batch containins a number of sequences divisible by `shard_multiplier`.

  If `drop_long_sequences` is True, then sequences longer than
  `hparams.batch_size` are dropped.  This prevents generating batches with
  more than the usual number of tokens, which can cause out-of-memory errors.

  Args:
    hparams: a hyperparameters.
    drop_long_sequences: a boolean.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.

  Returns:
     a dictionary
  """
  max_length = hparams.max_length or hparams.batch_size
  boundaries = bucket_boundaries(
      max_length, mantissa_bits=hparams.batching_mantissa_bits)
  batch_sizes = [
      max(1, hparams.batch_size // length)
      for length in boundaries + [max_length]
  ]
  batch_sizes = [b * shard_multiplier for b in batch_sizes]
  max_length *= length_multiplier
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  return {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "max_length": (max_length if drop_long_sequences else 10**9)
  }


def get_datasets(problems, data_dir, mode):
  """Return the location of a dataset for a given mode."""
  datasets = []
  path = os.path.join(data_dir, problems)
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    datasets.append("%s-train*" % path)
  else:
    datasets.append("%s-dev*" % path)
  return datasets
