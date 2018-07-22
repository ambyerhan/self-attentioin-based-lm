#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tempfile

# Dependency imports

import numpy as np

from nlm.data_generators import generator_utils
from nlm.data_generators import ptb

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "", "Data directory.")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen", "Temporary storage directory.")
flags.DEFINE_integer("vocab_size", -1, "Size of the vocabulary [0 for staying all the words]")
flags.DEFINE_integer("num_shards", 10, "How many shards to use.")
flags.DEFINE_integer("random_seed", 429459, "Random seed to use.")

FILE_NAME_LIST=[
  '../datatmp/ptb.train.txt',
  '../datatmp/ptb.valid.txt',
  '../datatmp/ptb.test.txt'
]

def set_random_seed():
  """Set the random seed from flag everywhere."""
  tf.set_random_seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  problem = "lmptb_10k"
  if not FLAGS.data_dir:
    FLAGS.data_dir = tempfile.gettempdir()
    tf.logging.warning("It is strongly recommended to specify --data_dir. "
                       "Data will be written to default data_dir=%s.", FLAGS.data_dir)

  tf.logging.info("Generating problem:\n  * %s\n" % (problem))

  set_random_seed()
  generate_data_for_problem(problem, flist = FILE_NAME_LIST)


def generate_data_for_problem(problem, flist):
  """Generate data for a problem in _SUPPORTED_PROBLEM_GENERATORS."""
  if FLAGS.vocab_size < 0:
      tf.logging.info("[Error] Please specify the vocabulary size!")
      exit()

  PREFIX = problem + generator_utils.UNSHUFFLED_SUFFIX
  cptb = ptb.CPTB(tmp_dir = FLAGS.tmp_dir, data_dir = FLAGS.data_dir, flist = flist, vsize = FLAGS.vocab_size)

  ''' gening train '''
  tf.logging.info("Generating training data for %s.", problem)
  train_gen = cptb.train_generator()
  train_output_files = generator_utils.train_data_filenames(PREFIX, FLAGS.data_dir, FLAGS.num_shards)
  generator_utils.generate_files(train_gen, train_output_files)

  ''' gening dev '''
  tf.logging.info("Generating development data for %s.", problem)
  dev_gen = cptb.valid_generator()
  dev_output_files = generator_utils.dev_data_filenames(PREFIX, FLAGS.data_dir, 1)
  generator_utils.generate_files(dev_gen, dev_output_files)

  ''' shuffling '''
  all_output_files = train_output_files + dev_output_files
  tf.logging.info("Shuffling data...")
  generator_utils.shuffle_dataset(all_output_files)


if __name__ == "__main__":
  tf.app.run()
