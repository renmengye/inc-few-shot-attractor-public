"""Attention attractor for logistic regression."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import weight_variable
from fewshot.models.attractors.attractor import Attractor
from fewshot.models.attractors.attractor import RegisterAttractor


@RegisterAttractor('proto_attn_attr')
class ProtoAttentionAttractor(Attractor):

  def __call__(self, fast_weights, is_training=True, reuse=None, **kwargs):
    """Applies attention attractor.

    Args:
      fast_weights: A tuple of two elements.
        - w_b: [D, K]. Weights of the logistic regression.
        - b_b: [K]. Bias of the logistic regression.

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
    dtype = h_b.dtype
    with tf.variable_scope("transfer_loss", reuse=reuse):
      w_class_b_reg = self.combine_wb(fast_weights[0], fast_weights[1])
      h_size_reg = int(w_class_b_reg.shape[0])
      h_size = int(w_class_a.shape[0])
      tau_qq = weight_variable(
          [],
          dtype=dtype,
          init_method='constant',
          init_param={'val': self.config.attn_attr_tau_init},
          name='tau_qq')
      h_attend_bias = weight_variable(
          [h_size_reg],
          dtype=dtype,
          init_method='truncated_normal',
          init_param={'stddev': 1e-2},
          wd=self.config.wd,  # wasn't there before.
          name='h_attend_bias')
      num_classes_a = int(w_class_a.shape[1])
      num_classes_b = int(w_class_b_reg.shape[1])
      assert self.config.mlp_hidden != 0
      w_kb = weight_variable([h_size, self.config.mlp_hidden],
                             init_method='truncated_normal',
                             dtype=dtype,
                             init_param={'stddev': self.config.mlp_init},
                             wd=self.config.wd,
                             name='w_kb')
      b_kb = weight_variable([self.config.mlp_hidden],
                             init_method='constant',
                             dtype=dtype,
                             init_param={'val': 0.0},
                             wd=self.config.wd,
                             name='b_kb')
      w_kb2 = weight_variable([self.config.mlp_hidden, h_size_reg],
                              init_method='truncated_normal',
                              dtype=dtype,
                              init_param={'stddev': self.config.mlp_init},
                              wd=self.config.wd,
                              name='w_kb2')
      b_kb2 = weight_variable([h_size_reg],
                              init_method='constant',
                              dtype=dtype,
                              init_param={'val': 0.0},
                              wd=self.config.wd,
                              name='b_kb2')
      w_class_a_mask = tf.cond(
          mask, self._get_mask_fn(w_class_a, y_sel, num_classes_a),
          lambda: w_class_a)
      k_b = tf.matmul(
          tf.tanh(tf.matmul(tf.transpose(w_class_a_mask), w_kb) + b_kb),
          w_kb2) + b_kb2
      self._k_b = k_b
      k_b = tf.transpose(k_b)
      k_b_mask = tf.cond(mask, self._get_mask_fn(k_b, y_sel, num_classes_a),
                         lambda: k_b)

      if self.config.old_and_new:
        attended_h = self._compute_protos_attend(
            num_classes_b,
            h_b,
            y_b - num_classes_a,
            tau_qq,
            h_attend_bias,
            k_b_mask,
            w_class_a_mask,
            is_training=is_training)
      else:
        attended_h = self._compute_protos_attend5_fix(
            num_classes_b,
            h_b,
            y_b,
            tau_qq,
            h_attend_bias,
            k_b_mask,
            w_class_a_mask,
            is_training=is_training)
      self.attended_h = attended_h
      self.h_b = h_b

      # Cache the value of the attended features.
      if self.config.cache_transfer_loss_var:
        self._transfer_loss_var = attended_h
        tloss_var_plh = tf.placeholder(
            dtype, [None, h_size_reg], name='transfer_loss_var_plh')
        self._transfer_loss_var_plh = tloss_var_plh
        attended_h = tloss_var_plh
    with tf.variable_scope("new_loss", reuse=reuse):
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

    # [D, K2] and [K2, D]
    dist = tf.reduce_sum(
        tf.square(w_class_b_reg - tf.transpose(attended_h)) * gamma_, [0])
    transfer_loss = tf.reduce_mean(dist)
    return transfer_loss

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

  def _compute_protos_attend(self,
                             nclasses,
                             h_train,
                             y_train,
                             tau_q,
                             h_attend_bias,
                             k_b,
                             w_class_a,
                             is_training=True):
    """Compute prototypes using attention.

    Args:
      nclasses: N-way classification.
      h_train: Support example features.
      y_train: Support example labels (class indices).
      tau_q: Temperature parameter for the
      h_attend_bias: Biases of the attention.
      k_b: Attractor bases.
      w_class_a: Base class weights.
    """
    protos = self._compute_protos(nclasses, h_train, y_train)
    self.protos = protos
    if is_training:
      protos = tf.nn.dropout(protos, keep_prob=0.9)
    protos_norm = self._normalize(protos, axis=1)
    w_class_a_norm = self._normalize(w_class_a, axis=0)
    h_dot_w = tf.matmul(protos_norm, w_class_a_norm)
    h_dot_w *= tau_q
    attend = tf.nn.softmax(h_dot_w)  # [K proto, K wa]
    sel_idx = tf.argmax(attend, axis=1)
    # [B, K] * [K, D] = [B, D]
    h_attend = tf.matmul(attend, k_b, transpose_b=True)
    if h_attend_bias is not None:
      h_attend += h_attend_bias
    # [K, D]
    return h_attend

  def _get_mask_fn(self, w, y_sel, num_classes_a):
    """Mask the base classes."""

    bin_mask = tf.reduce_sum(
        tf.one_hot(y_sel, num_classes_a, dtype=w.dtype), 0, keep_dims=True)

    def mask_fn():
      w_m = w * (1.0 - bin_mask) + 1e-7 * bin_mask
      return w_m

    return mask_fn
