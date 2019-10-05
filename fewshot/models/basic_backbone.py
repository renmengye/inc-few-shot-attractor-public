"""A basic feature extractor.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.backbone import Backbone
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import cnn
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel('basic_backbone')
class BasicBackbone(Backbone):
  """A standard 4-layer conv net"""

  def __call__(self, x, is_training=True, ext_wts=None, reuse=None, **kwargs):
    """See Backbone class for documentation."""
    config = self.config
    dtype = self.dtype
    assert not config.add_last_relu
    L = len(config.conv_act_fn)
    if config.add_last_relu:
      act_fn = [tf.nn.relu for aa in range(L)]
    else:
      act_fn = [tf.nn.relu for aa in range(L - 1)] + [None]
    with tf.variable_scope("phi", reuse=reuse):
      assert config.wd == 5e-4, '{}'.format(config.wd)
      h, wts = cnn(
          x,
          config.filter_size,
          strides=config.strides,
          pool_fn=[tf.nn.max_pool] * len(config.pool_fn),
          pool_size=config.pool_size,
          pool_strides=config.pool_strides,
          act_fn=act_fn,
          add_bias=False,
          init_std=config.conv_init_std,
          init_method=config.conv_init_method,
          wd=config.wd,
          dtype=dtype,
          batch_norm=True,
          is_training=is_training,
          ext_wts=ext_wts)
      if self.weights is None:
        self.weights = wts
      h_shape = h.get_shape()
      h_size = 1
      for ss in h_shape[1:]:
        h_size *= int(ss)

      if ext_wts is not None:
        # PyTorch NCHW mode.
        h = tf.transpose(h, [0, 3, 1, 2])
      h = tf.reshape(h, [-1, h_size])
      assert h_size == 3200
    return h
