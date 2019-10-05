"""Static attractor for MLP (with residual connection)."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import weight_variable
from fewshot.models.attractors.attractor import Attractor
from fewshot.models.attractors.attractor import RegisterAttractor


@RegisterAttractor('static_attr_resmlp')
class StaticResMLPAttractor(Attractor):

  def __call__(self, fast_weights, reuse=None, **kwargs):
    """Applies static attractor.

    Args:
      fast_weights: A tuple of the following elements:
        w_1: [D, K]. Logistic regression weights.
        w_2: [D, H]. First layer weights.
        b_2: [H]. First layer biases.
        w_3: [H, K]. Second layer weights.
        b_3: [K]. Second layer bias.

      reuse: Bool. Whether to reuse variables.
    """
    with tf.variable_scope("transfer_loss", reuse=reuse):
      w_1 = fast_weights[0]
      w_2 = fast_weights[1]
      b_2 = fast_weights[2]
      w_3 = fast_weights[3]
      b_3 = fast_weights[4]
      dtype = w_1.dtype
      Hplus1 = int(w_3.shape[0]) + 1  # H+1
      Dplus1 = int(w_2.shape[0]) + 1  # D+1
      # Logistic regression weights + biases.
      w_class_b_reg = self.combine_wb(w_3, b_3)  # [H+1, K]
      # First layer weights + biases.
      w_class_b_reg2 = self.combine_wb(w_2, b_2)  # [D+1, H]
      # Second layer weights.
      b_1 = tf.zeros([int(w_1.shape[1])], dtype=dtype)
      w_class_b_reg3 = self.combine_wb(w_1, b_1)  # [D+1, K]
      attr = weight_variable([Hplus1],
                             init_method='truncated_normal',
                             dtype=dtype,
                             init_param={'stddev': 0.01},
                             wd=self.config.wd,
                             name='attr')
      attr2 = weight_variable([Dplus1],
                              init_method='truncated_normal',
                              dtype=dtype,
                              init_param={'stddev': 0.01},
                              wd=self.config.wd,
                              name='attr2')
      attr3 = weight_variable([Dplus1],
                              init_method='truncated_normal',
                              dtype=dtype,
                              init_param={'stddev': 0.01},
                              wd=self.config.wd,
                              name='attr3')
      attr_ = tf.expand_dims(attr, 1)  # [H+1, 1]
      attr2_ = tf.expand_dims(attr2, 1)  # [D+1, 1]
      attr3_ = tf.expand_dims(attr3, 1)  # [D+1, 1]

      init_log_gamma = np.log(self.config.init_gamma)
      if self.config.learn_gamma:
        log_gamma = weight_variable([Hplus1],
                                    init_method='constant',
                                    dtype=dtype,
                                    init_param={'val': init_log_gamma},
                                    wd=self.config.wd,
                                    name='log_gamma')
        log_gamma2 = weight_variable([Dplus1],
                                     init_method='constant',
                                     dtype=dtype,
                                     init_param={'val': init_log_gamma},
                                     wd=self.config.wd,
                                     name='log_gamma2')
        log_gamma3 = weight_variable([Dplus1],
                                     init_method='constant',
                                     dtype=dtype,
                                     init_param={'val': init_log_gamma},
                                     wd=self.config.wd,
                                     name='log_gamma3')
      else:
        log_gamma = tf.ones([Hplus1], dtype=dtype) * init_log_gamma
        log_gamma2 = tf.ones([Dplus1], dtype=dtype) * init_log_gamma
        log_gamma3 = tf.ones([Dplus1], dtype=dtype) * init_log_gamma
      gamma_ = tf.exp(tf.expand_dims(log_gamma, 1))  # [H+1, 1]
      gamma2_ = tf.exp(tf.expand_dims(log_gamma2, 1))  # [D+1, 1]
      gamma3_ = tf.exp(tf.expand_dims(log_gamma3, 1))  # [D+1, 1]
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(w_class_b_reg - attr_) * gamma_, [0]))
    loss += tf.reduce_mean(
        tf.reduce_sum(tf.square(w_class_b_reg2 - attr2_) * gamma2_, [0]))
    loss += tf.reduce_mean(
        tf.reduce_sum(tf.square(w_class_b_reg3 - attr3_) * gamma3_, [0]))
    return loss
