"""Static attractor for logistic regression."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import weight_variable
from fewshot.models.attractors.attractor import Attractor
from fewshot.models.attractors.attractor import RegisterAttractor


@RegisterAttractor('static_attr')
class StaticAttractor(Attractor):

  def __call__(self, fast_weights, reuse=None, **kwargs):
    """Applies static attractor.

    Args:
      fast_weights: A tuple of two elements.
          - w_b: [D, K]. Weights of the logistic regression.
          - b_b: [K]. Bias of the logistic regression.
      reuse: Bool. Whether to reuse variables.
    """
    with tf.variable_scope("transfer_loss", reuse=reuse):
      w_class_b_reg = self.combine_wb(fast_weights[0], fast_weights[1])
      dtype = fast_weights[0].dtype
      h_size_reg = int(w_class_b_reg.shape[0])
      attr = weight_variable([h_size_reg],
                             dtype=dtype,
                             init_method='constant',
                             init_param={'val': 0.0},
                             wd=self.config.wd,
                             name='attr')
      attr_ = tf.expand_dims(attr, 1)
      if self.config.learn_gamma:
        log_gamma = weight_variable(
            [h_size_reg],
            init_method='constant',
            dtype=dtype,
            init_param={'val': np.log(self.config.init_gamma)},
            wd=self.config.wd,
            name='log_gamma')
      else:
        log_gamma = tf.ones([h_size_reg], dtype=dtype) * np.log(
            self.config.init_gamma)
      log_gamma_ = tf.expand_dims(log_gamma, 1)
      gamma_ = tf.exp(log_gamma_)
      self.gamma = gamma_
    dist = tf.reduce_sum(tf.square(w_class_b_reg - attr_) * gamma_, [0])
    transfer_loss = tf.reduce_mean(dist)
    return transfer_loss
