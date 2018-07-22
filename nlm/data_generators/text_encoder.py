
"""Encoders for text data.

* TextEncoder: base class
* ByteTextEncoder: for ascii text
* TokenTextEncoder: with user-supplied vocabulary file
* SubwordTextEncoder: invertible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

# Dependency imports

import six
from six import PY2
from six import unichr  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Reserved tokens for things like padding and EOS symbols.
PAD = "PAD"
UNK = "UNK"
EOS = "EOS"
RESERVED_TOKENS = [PAD, UNK, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_TOKEN = RESERVED_TOKENS.index(PAD)  # Normally 0
UNK_TOKEN = RESERVED_TOKENS.index(UNK)  # Normally 1
EOS_TOKEN = RESERVED_TOKENS.index(EOS)  # Normally 2
RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(UNK, "ascii"), bytes(EOS, "ascii")]


native_to_unicode = lambda s: s
unicode_to_native = lambda s: s


class TextEncoder(object):
  """Base class for converting from ints to/from human readable strings."""

  def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
    self._num_reserved_ids = num_reserved_ids

  @property
  def num_reserved_ids(self):
    return self._num_reserved_ids

  def encode(self, s):
    """Transform a human-readable string into a sequence of int ids.

    The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
    num_reserved_ids) are reserved.

    EOS is not appended.

    Args:
      s: human-readable string to be converted.

    Returns:
      ids: list of integers
    """
    return [int(w) + self._num_reserved_ids for w in s.split()]

  def decode(self, ids):
    """Transform a sequence of int ids into a human-readable string.

    EOS is not expected in ids.

    Args:
      ids: list of integers to be converted.

    Returns:
      s: human-readable string.
    """
    decoded_ids = []
    for id_ in ids:
      if 0 <= id_ < self._num_reserved_ids:
        decoded_ids.append(RESERVED_TOKENS[int(id_)])
      else:
        decoded_ids.append(id_ - self._num_reserved_ids)
    return " ".join([str(d) for d in decoded_ids])

  @property
  def vocab_size(self):
    raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
  """Encoder based on a user-supplied vocabulary."""

  def __init__(self, vocab_filename, reverse=False,
               num_reserved_ids=NUM_RESERVED_TOKENS):
    """Initialize from a file, one token per line."""
    super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
    self._reverse = reverse
    self._load_vocab_from_file(vocab_filename)

  def encode(self, sentence):
    """Converts a space-separated string of tokens to a list of ids."""
    ret = [self._token_to_id.get(tok, UNK_TOKEN) for tok in sentence.strip().split()]
    return ret[::-1] if self._reverse else ret

  def decode(self, ids):
    seq = reversed(ids) if self._reverse else ids
    return " ".join([self._safe_id_to_token(i) for i in seq])

  @property
  def vocab_size(self):
    return len(self._id_to_token)

  def _safe_id_to_token(self, idx):
    return self._id_to_token.get(idx, "ID_%d" % idx)

  def _load_vocab_from_file(self, filename):
    """Load vocab from a file."""
    self._token_to_id = {}
    self._id_to_token = {}

    token_start_idx = 0
    with tf.gfile.Open(filename) as f:
      for i, line in enumerate(f):
        idx = token_start_idx + i
        tok = line.strip()
        self._token_to_id[tok] = idx
        self._id_to_token[idx] = tok
