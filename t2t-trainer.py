#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys

# Dependency imports

from nlm.utils import trainer_utils as utils

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("t2t_usr_dir", "",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  utils.validate_flags()
  utils.run(
      data_dir=FLAGS.data_dir,
      model=FLAGS.model,
      output_dir=FLAGS.output_dir,
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.app.run()
