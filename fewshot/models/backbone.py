"""Feature extractor backbone interface.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf


class Backbone(object):
  """Defines the interface of a backbone network,"""

  def __init__(self, config, dtype=tf.float32):
    """Constructs a backbone object with some network configurations."""
    self._config = config
    self._dtype = dtype
    self._weights = None

  def __call__(self, x, is_training=True, ext_wts=None, reuse=None):
    """Extract features from raw inputs.

    Args:
      x: [N, H, W, C]. Inputs.
      ext_wts: Whether to use externally provided weights.
      reuse: Whether to reuse weights, default None.
    """
    raise NotImplemented()

  def get_weights_dict(self):
    raise NotImplemented()

  @property
  def config(self):
    return self._config

  @property
  def weights(self):
    return self._weights

  @weights.setter
  def weights(self, weights):
    self._weights = weights

  @property
  def dtype(self):
    return self._dtype
