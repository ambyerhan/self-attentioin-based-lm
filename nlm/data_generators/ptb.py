
"""Data generators for PTB data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tarfile

# Dependency imports

from nlm.data_generators import generator_utils
from nlm.data_generators import text_encoder

import tensorflow as tf

PAD = text_encoder.PAD
UNK = text_encoder.UNK
EOS = text_encoder.EOS

def _read_words(filename):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    assert sys.version_info[0] >= 3
    return f.read().replace("\n", " ").split()


def _build_vocab(filename, vocab_path, vocab_size):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  words = words if vocab_size == 0 else words[:vocab_size - 3] # vsize == 0 : stay all the words
  with open(vocab_path, "w") as f:
    f.write("%s\n" % PAD)
    f.write("%s\n" % UNK)
    f.write("%s\n" % EOS)
    f.write("\n".join(words))
  tf.logging.info("The vocab size = %d" % (len(words) + 3))


def _get_token_encoder(vocab_dir, filename, vocab_size):
  """Reads from file and returns a `TokenTextEncoder` for the vocabulary."""
  vocab_name = "lmptb_10k.vocab"
  vocab_path = os.path.join(vocab_dir, vocab_name)
  _build_vocab(filename, vocab_path, vocab_size)
  return text_encoder.TokenTextEncoder(vocab_path)


class CPTB(object):
  """A class for generating PTB data."""

  def __init__(self, tmp_dir, data_dir, flist, vsize):

    files = flist
    for filename in files:
      if "train" in filename:
        self.train = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        self.valid = os.path.join(tmp_dir, filename)

    assert hasattr(self, "train"), "Training file not found"
    assert hasattr(self, "valid"), "Validation file not found"
    self.encoder = _get_token_encoder(data_dir, self.train, vocab_size = vsize)

  def train_generator(self):
    return self._generator(self.train)

  def valid_generator(self):
    return self._generator(self.valid)

  def _generator(self, filename):
    with tf.gfile.GFile(filename, "r") as f:
      for line in f:
        line = " ".join(line.replace("\n", EOS).split())
        tok = self.encoder.encode(line)
        yield {"inputs": tok[:-1], "targets": tok} # targets: w1, w2, ..., wn, eos    when training, it become input: pad, w1, w2, ..., wn; output: w1, w2, ..., wn, eos

