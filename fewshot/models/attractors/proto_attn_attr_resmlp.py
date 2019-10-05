"""Attention attractor for MLP with residual connection."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import weight_variable
from fewshot.models.attractors.attractor import Attractor
from fewshot.models.attractors.attractor import RegisterAttractor


@RegisterAttractor('proto_attn_attr_resmlp')
class ProtoAttentionResMLPAttractor(Attractor):

  def __call__(self, fast_weights, is_training=True, reuse=None, **kwargs):
    """Applies attention attractor for MLP with residual connection.

    Args:
      fast_weights: A tuple of the following elements:
        w_1: [D, K]. Logistic regression weights.
        w_2: [D, H]. First layer weights.
        b_2: [H]. First layer biases.
        w_3: [H, K]. Second layer weights.
        b_3: [K]. Second layer bias.

      reuse: Bool. Whether to reuse variables.

      kwargs: Contains the following fields:
        - y_b: Labels of the support examples.
        - h_b: Features of the support examples.
        - w_class_a: Base class weights.
        - mask: Bool flag whether we need to mask the base class weights.
        - y_sel: Binary flag on the base class weights.
    """
    y_b = kwargs['y_b']
    h_b = kwargs['h_b']
    w_class_a = kwargs['w_class_a']
    mask = kwargs['mask']
    y_sel = kwargs['y_sel']
    with tf.variable_scope("transfer_loss", reuse=reuse):
      w_1 = fast_weights[0]
      w_2 = fast_weights[1]
      b_2 = fast_weights[2]
      w_3 = fast_weights[3]
      b_3 = fast_weights[4]
      dtype = w_1.dtype
      Hplus1 = int(w_3.shape[0]) + 1  # H+1
      Dplus1 = int(w_2.shape[0]) + 1  # D+1
      K = int(w_1.shape[1])
      D = int(w_1.shape[0])
      M = self.config.mlp_hidden
      # Logistic regression weights + biases.
      w_class_b_reg = self.combine_wb(w_3, b_3)  # [H+1, K]
      # First layer weights + biases.
      w_class_b_reg2 = self.combine_wb(w_2, b_2)  # [D+1, H]
      # Second layer weights.
      b_1 = tf.zeros([int(w_1.shape[1])], dtype=dtype)
      w_class_b_reg3 = self.combine_wb(w_1, b_1)  # [D+1, K]
      h_size_reg = int(w_class_b_reg.shape[0])  # H+1
      h_size_reg2 = int(w_class_b_reg2.shape[0])  # D+1
      h_size = int(w_class_a.shape[0])  # D
      tau_init = self.config.attn_attr_tau_init
      tau_q = weight_variable([],
                              init_method='constant',
                              dtype=dtype,
                              init_param={'val': tau_init},
                              name='tau_qq')
      tau_q2 = weight_variable([],
                               init_method='constant',
                               dtype=dtype,
                               init_param={'val': tau_init},
                               name='tau_qq2')
      Ko = int(w_class_a.shape[1])  # Kold
      h_attend_bias = weight_variable([Hplus1],
                                      dtype=dtype,
                                      init_method='truncated_normal',
                                      init_param={'std': 1e-2},
                                      wd=self.config.wd,
                                      name='h_attend_bias')
      h_attend_bias2 = weight_variable([Dplus1],
                                       dtype=dtype,
                                       init_method='truncated_normal',
                                       init_param={'std': 1e-2},
                                       wd=self.config.wd,
                                       name='h_attend_bias2')
      h_attend_bias3 = weight_variable([Dplus1],
                                       dtype=dtype,
                                       init_method='truncated_normal',
                                       init_param={'std': 1e-2},
                                       wd=self.config.wd,
                                       name='h_attend_bias3')
      assert self.config.mlp_hidden != 0
      w_kb = weight_variable([D, M],
                             init_method='truncated_normal',
                             dtype=dtype,
                             init_param={'stddev': self.config.mlp_init},
                             wd=self.config.wd,
                             name='w_kb')
      b_kb = weight_variable([M],
                             init_method='constant',
                             dtype=dtype,
                             init_param={'val': 0.0},
                             wd=self.config.wd,
                             name='b_kb')
      w_kb21 = weight_variable([M, Hplus1],
                               init_method='truncated_normal',
                               dtype=dtype,
                               init_param={'stddev': self.config.mlp_init},
                               wd=self.config.wd,
                               name='w_kb21')
      b_kb21 = weight_variable([Hplus1],
                               init_method='constant',
                               dtype=dtype,
                               init_param={'val': 0.0},
                               wd=self.config.wd,
                               name='b_kb21')
      w_kb22 = weight_variable([M, 2 * Dplus1],
                               init_method='truncated_normal',
                               dtype=dtype,
                               init_param={'stddev': self.config.mlp_init},
                               wd=self.config.wd,
                               name='w_kb22')
      b_kb22 = weight_variable([2 * Dplus1],
                               init_method='constant',
                               dtype=dtype,
                               init_param={'val': 0.0},
                               wd=self.config.wd,
                               name='b_kb22')

      w_class_a_mask = tf.cond(mask, self._get_mask_fn(w_class_a, y_sel, Ko),
                               lambda: w_class_a)
      # [Ko, D+1] -> [Ko, M]
      kbz = tf.tanh(tf.matmul(tf.transpose(w_class_a_mask), w_kb) + b_kb)
      # [Ko, M] -> [Ko, H+1]
      k_b = tf.matmul(kbz, w_kb21) + b_kb21
      # [Ko, M] -> [Ko, 2(D+1)]
      k_b2 = tf.matmul(kbz, w_kb22) + b_kb22
      k_b = tf.transpose(k_b)  # [H+1, Ko]
      k_b2 = tf.transpose(k_b2)  # [2(D+2), Ko]
      k_b_mask = tf.cond(mask, self._get_mask_fn(k_b, y_sel, Ko), lambda: k_b)
      k_b2_mask = tf.cond(mask, self._get_mask_fn(k_b2, y_sel, Ko),
                          lambda: k_b2)

      if self.config.old_and_new:
        y_b = y_b - Ko
      protos = self._compute_protos(K, h_b, y_b)  # [K, D]
      if is_training:
        protos = tf.nn.dropout(protos, keep_prob=0.9)
      protos_norm = self._normalize(protos, axis=1)  # [K, D]
      episode_mean = tf.reduce_mean(h_b, [0], keepdims=True)  # [1, D]
      episode_norm = self._normalize(episode_mean, axis=1)  # [1, D]
      w_class_a_norm = self._normalize(w_class_a_mask, axis=0)  # [D, Ko]
      h_dot_w = tf.matmul(protos_norm, w_class_a_norm)  # [K, Ko]
      e_dot_w = tf.matmul(episode_norm, w_class_a_norm)  # [1, Ko]
      h_dot_w *= tau_q  # [K, Ko]
      e_dot_w *= tau_q2  # [1, Ko]
      proto_attend = tf.nn.softmax(h_dot_w)  # [K, Ko]
      episode_attend = tf.nn.softmax(e_dot_w)  # [1, Ko]
      k_b3 = k_b2[h_size_reg2:, :]  # [D+1, Ko]
      k_b2 = k_b2[:h_size_reg2, :]  # [D+1, Ko]
      h_attend = tf.matmul(proto_attend, k_b, transpose_b=True)  # [K, H+1]
      h_attend2 = tf.matmul(episode_attend, k_b2, transpose_b=True)  # [1, D+1]
      h_attend3 = tf.matmul(proto_attend, k_b3, transpose_b=True)  # [K, D+1]
      attr_ = tf.transpose(h_attend + h_attend_bias)  # [K, H+1] -> [H+1, K]
      attr2_ = tf.transpose(h_attend2 + h_attend_bias2)  # [1, D+1] -> [D+1, 1]
      attr3_ = tf.transpose(h_attend3 + h_attend_bias3)  # [K, D+1] -> [D+1, K]
    with tf.variable_scope("new_loss", reuse=reuse):
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

  def _normalize(self, x, axis, eps=1e-5):
    """Normalize a vector (for calculating cosine similarity."""
    return x / (
        tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)) + 1e-5)

  def _compute_protos(self, nclasses, h_train, y_train):
    """Computes the prototypes, cluster centers.
    Args:
      nclasses: Int. Number of classes.
      h_train: [B, N, D], Train features.
      y_train: [B, N], Train class labels.
    Returns:
      protos: [B, K, D], Test prediction.
    """
    protos = [None] * nclasses
    for kk in range(nclasses):
      # [N, 1]
      ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype), 1)
      # [N, D]
      protos[kk] = tf.reduce_sum(h_train * ksel, [0], keep_dims=True)
      protos[kk] /= (tf.reduce_sum(ksel, [0, 1], keep_dims=True) + 1e-7)
    protos = tf.concat(protos, axis=0)  # [K, D]
    return protos

  def _get_mask_fn(self, w, y_sel, num_classes_a):
    """Mask the base classes."""

    bin_mask = tf.reduce_sum(
        tf.one_hot(y_sel, num_classes_a, dtype=w.dtype), 0, keep_dims=True)

    def mask_fn():
      w_m = w * (1.0 - bin_mask) + 1e-7 * bin_mask
      return w_m

    return mask_fn
