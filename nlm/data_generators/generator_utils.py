
"""Utilities for data generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import io
import os
import random

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin


import tensorflow as tf

UNSHUFFLED_SUFFIX = "-unshuffled"


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s", str((k, v)))
    if isinstance(v[0], six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))


def _data_filenames(output_name, output_dir, num_shards):
  return [
      os.path.join(output_dir, fname)
      for fname in shard_filepath(output_name, num_shards)
  ]


def train_data_filenames(problem, output_dir, num_shards):
  return _data_filenames(problem + "-train", output_dir, num_shards)


def dev_data_filenames(problem, output_dir, num_shards):
  return _data_filenames(problem + "-dev", output_dir, num_shards)


def test_data_filenames(problem, output_dir, num_shards):
  return _data_filenames(problem + "-test", output_dir, num_shards)


def sharded_name(base_name, shard, total_shards):
  return "%s-%.5d-of-%.5d" % (base_name, shard, total_shards)


def shard_filepath(fname, num_shards):
  return [sharded_name(fname, shard, num_shards) for shard in xrange(num_shards)]


def generate_files(generator, output_filenames, max_cases=None):
  num_shards = len(output_filenames)
  writers = [tf.python_io.TFRecordWriter(fname) for fname in output_filenames]
  counter, shard = 0, 0
  for case in generator:
    if counter > 0 and counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writers[shard].write(sequence_example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()


def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("read: %d", len(records))
  return records


def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
    if count > 0 and count % 100000 == 0:
      tf.logging.info("write: %d", count)
  writer.close()


def generate_dataset_and_shuffle(train_gen, train_paths, dev_gen, dev_paths, shuffle=True):
  generate_files(train_gen, train_paths)
  generate_files(dev_gen, dev_paths)
  if shuffle:
    shuffle_dataset(train_paths + dev_paths)


def shuffle_dataset(filenames):
  tf.logging.info("Shuffling data...")
  for fname in filenames:
    records = read_records(fname)
    random.shuffle(records)
    out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
    write_records(records, out_fname)
    tf.gfile.Remove(fname)
