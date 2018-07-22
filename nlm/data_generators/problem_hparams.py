

"""Hyperparameters defining different problems.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from nlm.data_generators import text_encoder
from nlm.models import modalities  # pylint: disable=unused-import

import tensorflow as tf


def default_problem_hparams():
  """A set of basic model hyperparameters."""
  return tf.contrib.training.HParams(
      # Use this parameter to get comparable perplexity numbers with different
      # tokenizations.  This value should be set to the ratio of the number of
      # tokens in the test set according to the tokeization used to the number
      # of tokens in the test set in the "official" tokenization.  For example,
      # if we are using a word-piece based model and we want to compute
      # per-word perplexity, then we set loss_multiplier to the number of
      # wordpieces per word in the test set.
      loss_multiplier=1.0,

      # Use this parameter to allow for larger sequences in the batch. Without
      # the use of this parameter, the size of the inner two dimensions will be
      # used to judge the sequence length.
      batch_size_multiplier=1,

      # To make queues of the right capacity, it's good to know the maximal
      # expected batch size, as it can vary a lot. It only affects performance
      # of input readers and memory use. The defaults should be safe and fast,
      # but decrease if your reader uses a lot of memory and increase if slow.
      max_expected_batch_size_per_shard=64,

      # Modalities used to map from input features to a space compatible with
      # chosen model architecture.  One modality spec (which is a 2-tuple,
      # (modality_full_name, vocab_size)) per feature key. modality_full_name is
      # a string type:name, e.g. class_label:class_label_2d. Leaving off the
      # name uses the default modality for that type (e.g. class_label ==
      # class_label:default).
      input_modality={},

      # Modality used to map from hidden representation to the target space.
      # Specified as a modality spec, a 2-tuple described above.
      target_modality=None,

      # Identifiers used to tell the model which input/target space will be
      # expected. For example, it can tell that we expect French as characters
      # as output, or Spanish as sound. An integer with the following semantics:
      #   0: Generic / unknown output space (default)
      #   1: Image labels
      #   2: English characters
      #   3: English tokens
      #   4: English bpe tokens
      #   5: French characters
      #   6: French tokens
      #   7: German characters
      #   8: German tokens
      #   9: German bpe tokens
      #   10: Digit cipher lexicon 0
      #   11: Digit cipher lexicon 1
      #   12: Audio waveform domain
      #   13: Audio spectral domain
      #   14: Parse characters
      #   15: Parse tokens
      #   16: Chinese tokens
      #   17: Icelandic characters
      #   18: Icelandic tokens
      #   19: Icelandic parse tokens
      # Add more above if needed.
      input_space_id=0,
      target_space_id=0,

      # Vocabulary per feature key.
      #   a vocabulary converts to/from human-readable strings.
      # E.g. {"inputs": text_encoder.ByteTextEncoder(),
      #       "targets": text_encoder.SubwordTextEncoder("vocab_filename.txt")}
      vocabulary={
          "inputs": text_encoder.TextEncoder(),
          "targets": text_encoder.TextEncoder()
      },

      # This is a marker to keep track if the problem was reversed or copied.
      # Only set automatically, do not override the default.
      #
      # These tags can be combined in order to perform copies of the input or
      # the targets. For instance `problem_copy` will copy the inputs, but
      # `problem_rev_copy` will copy the targets.
      was_reversed=False,
      was_copy=False,
  )


def lm1b_32k(model_hparams):
  """Billion-word language-modeling benchmark, 32k subword vocabulary."""
  p = default_problem_hparams()
  # ratio of dev tokens (including eos) to dev words (including eos)
  # 176884 / 159658 = 1.107893
  p.perplexity_exponent = 1.107893
  p.input_modality = {}
  encoder = text_encoder.SubwordTextEncoder(os.path.join(model_hparams.data_dir, "lm1b_32k.subword_text_encoder"))
  p.target_modality = ("symbol", encoder.vocab_size)
  p.vocabulary = {"targets": encoder}
  p.target_space_id = 3
  return p


def lmptb_10k(model_hparams):
  """Penn Tree Bank language-modeling benchmark, 10k token vocabulary."""
  p = default_problem_hparams()
  p.input_modality = {}
  vocabulary = text_encoder.TokenTextEncoder(os.path.join(model_hparams.data_dir, "lmptb_10k.vocab"))
  p.target_modality = ("symbol", 10000)
  p.vocabulary = {"targets": vocabulary}
  p.input_space_id = 3
  p.target_space_id = 3
  return p

