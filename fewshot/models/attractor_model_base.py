"""A common abstract class for attractor based models.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import weight_variable
from fewshot.utils.logger import get as get_logger
from fewshot.models.attractors.attractor import get_attractor
from fewshot.models.attractors.static_attr import StaticAttractor  # NOQA
from fewshot.models.attractors.proto_attn_attr import ProtoAttentionAttractor  # NOQA
from fewshot.models.attractors.static_attr_resmlp import StaticResMLPAttractor  # NOQA
from fewshot.models.attractors.proto_attn_attr_resmlp import ProtoAttentionResMLPAttractor  # NOQA
log = get_logger()


class AttractorModelBase(object):
  """Base class for attractor models."""

  def build_task_a(self, x, y, is_training, ext_wts=None):
    """Build task A branch.

    Args:
      x: Tensor. [N, H, W, C]. Inputs tensor.
      y: Tensor. [N]. Labels tensor.
      is_training: Bool. Whether in training mode.
      ext_wts: Dict. External weights dictionary.
      opt: Optimizer object.
    """
    config = self.config
    global_step = self.global_step
    if config.backbone_class == 'resnet_backbone':
      bb_config = config.resnet_config
    else:
      assert False, 'Not supported'
    proto_config = config.protonet_config
    opt_config = config.optimizer_config
    num_classes_a = self._num_classes_a

    # Classification branch for task A.
    h_a = self._run_backbone(x, is_training=is_training, ext_wts=ext_wts)
    self._h_a = h_a
    h_shape = h_a.get_shape()
    h_size = 1
    for ss in h_shape[1:]:
      h_size *= int(ss)
    self._h_size = h_size

    if ext_wts is not None:
      w_class_a = weight_variable(
          [h_size, num_classes_a],
          init_method='numpy',
          dtype=self.dtype,
          init_param={'val': np.transpose(ext_wts['w_class_a'])},
          wd=bb_config.wd,
          name='w_class_a')
      b_class_a = weight_variable([],
                                  init_method='numpy',
                                  dtype=self.dtype,
                                  init_param={'val': ext_wts['b_class_a']},
                                  wd=0e0,
                                  name='b_class_a')
    else:
      w_class_a = weight_variable([h_size, num_classes_a],
                                  init_method='truncated_normal',
                                  dtype=self.dtype,
                                  init_param={'stddev': 0.01},
                                  wd=bb_config.wd,
                                  name='w_class_a')
      b_class_a = weight_variable([num_classes_a],
                                  dtype=self.dtype,
                                  init_method='constant',
                                  init_param={'val': 0.0},
                                  name='b_class_a')
    self._w_class_a = w_class_a
    self._b_class_a = b_class_a
    num_classes_a_dyn = tf.cast(tf.shape(b_class_a)[0], tf.int64)
    num_classes_a_dyn32 = tf.shape(b_class_a)[0]

    if proto_config.cosine_a:
      if proto_config.cosine_tau:
        if ext_wts is None:
          tau_init_val = 10.0
        else:
          tau_init_val = ext_wts['tau'][0]
        tau = weight_variable([],
                              dtype=self.dtype,
                              init_method='constant',
                              init_param={'val': tau_init_val},
                              name='tau')
      else:
        tau = tf.constant(1.0)

      w_class_a_norm = self._normalize(w_class_a, axis=0)
      h_a_norm = self._normalize(h_a, axis=1)
      dot = tf.matmul(h_a_norm, w_class_a_norm)
      if ext_wts is not None:
        dot += b_class_a
      logits_a = tau * dot
    else:
      logits_a = tf.matmul(h_a, w_class_a) + b_class_a

    self._prediction_a = logits_a
    self._prediction_a_all = self._prediction_a
    y_dense = tf.one_hot(y, num_classes_a)
    xent_a = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_a, labels=y_dense)
    xent_a = tf.reduce_mean(xent_a, name='xent')
    cost_a = xent_a
    self._cost_a = cost_a
    cost_a += self._decay()
    self._prediction_a = logits_a
    return logits_a

  def build_task_a_grad(self):
    # Gradients for Task A for all trainable variables.
    cost_a = self._cost_a
    var_list_a = tf.trainable_variables()
    grads_and_vars_a = list(zip(tf.gradients(cost_a, var_list_a), var_list_a))
    return grads_and_vars_a

  def _run_backbone(self, x, ext_wts=None, reuse=None, is_training=True):
    if self.config.backbone_class.startswith('resnet'):
      return self.backbone(
          x,
          is_training=is_training,
          ext_wts=ext_wts,
          reuse=reuse,
          slow_bn=ext_wts is not None)
    else:
      return self.backbone(x, is_training=is_training, ext_wts=ext_wts)

  def initialize(self, sess):
    # sess.run(self._initializer, feed_dict=self._init_fdict)
    sess.run(self._initializer)

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

  def _merge_var_list(self, vdict, vlist_old, vlist_new):
    if vdict is None:
      return None
    vdict_new = dict(list(vdict.items()))
    for vvo, vvn in zip(vlist_old, vlist_new):
      vname = vvo.name.split(':')[0]
      assert vname in vdict, '{} not found'.format(vname)
      vdict_new[vname] = vvn
    return vdict_new

  def _aggregate_grads_and_vars(self, grads_and_vars_list, weights=None):
    """Aggregates two sets of gradients by doing an weighted sum."""
    aggregated = {}
    log.info('Number of grads and vars to aggregate: {}'.format(
        len(grads_and_vars_list)))
    if weights is None:
      assert False, 'Equally aggregated, debug point'
      weights = [None] * len(grads_and_vars_list)
    for gv_list, wt in zip(grads_and_vars_list, weights):
      for g, v in gv_list:
        if g is not None:
          if v in aggregated:
            log.info('Variable matched in the dictionary: {}'.format(v.name))
            if wt is None:
              aggregated[v].append(g)
              log.info('Applied default weight 1.0')
            else:
              aggregated[v].append(g * wt)
              log.info('Applied weight {}'.format(wt))
          else:
            log.info('Variable created in the dictionary: {}'.format(v.name))
            if wt is None:
              aggregated[v] = [g]
              log.info('Applied default weight 1.0')
            else:
              aggregated[v] = [g * wt]
              log.info('Applied weight {}'.format(wt))
    result = []
    for v in aggregated.keys():
      log.info('Variable {} Count {}'.format(v.name, len(aggregated[v])))
      aggregated[v] = tf.add_n(aggregated[v])
      result.append((aggregated[v], v))
    return result

  def _get_mask_fn(self, w, num_classes_a):

    bin_mask = tf.reduce_sum(
        tf.one_hot(self._y_sel, num_classes_a, dtype=self.dtype),
        0,
        keep_dims=True)

    def mask_fn():
      w_m = w * (1.0 - bin_mask) + 1e-7 * bin_mask
      return w_m

    return mask_fn

  def _apply_transfer_loss(self, fast_weights, reuse=None, **kwargs):
    """Apply fast loss.
    Args:
      fast_weights: Fast weights to optimize in the inner loop.
      reuse: Bool. Whether to reuse variables.

    Returns:
      loss: Scalar. Fast weights loss.
    """
    config = self.config
    tconfig = self.config.transfer_config
    bb_config = self.config.resnet_config
    loss_type = tconfig.transfer_loss_type

    def get_arg_or_default(key, default):
      return kwargs[key] if key in kwargs else default

    h_b = get_arg_or_default('h_b', None)
    y_b = get_arg_or_default('y_b', None)
    w_class_a = get_arg_or_default('w_class_a', None)
    b_class_a = get_arg_or_default('b_class_a', None)
    kwargs['mask'] = self._mask  # TODO: consider inject this elsewhere.
    kwargs['y_sel'] = self._y_sel

    def combine_wb(w, b):
      return tf.concat([w, tf.expand_dims(b, 0)], axis=0)

    attractor = get_attractor(loss_type, tconfig)
    self._attractor = attractor
    if attractor is not None:
      return attractor(
          fast_weights, is_training=self._is_training, reuse=reuse, **kwargs)
    else:
      assert False, loss_type

  def _normalize(self, x, axis, eps=1e-5):
    """Normalize a vector (for calculating cosine similarity."""
    return x / (
        tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)) + 1e-5)

  def _decay(self):
    """Weight decay for slow weights."""
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info('Weight decay variables')
    [log.info(x) for x in wd_losses]
    log.info('Total length: {}'.format(len(wd_losses)))
    if len(wd_losses) > 0:
      return tf.add_n(wd_losses)
    else:
      log.warning('No weight decay variables!')
      return 0.0

  def _decay_list(self, var_list):
    if len(var_list) > 0:
      wd = self.config.transfer_config.wd
      wd_losses = list(
          map(lambda x: 0.5 * wd * tf.reduce_sum(tf.square(x)), var_list))
      log.info('Weight decay variables')
      [log.info(x) for x in wd_losses]
      return tf.add_n(wd_losses)
    else:
      log.warning('No weight decay variables!')
      return 0.0

  def _ft_decay(self, var_list):
    """Weight decay for fast weights."""
    if len(var_list) > 0:
      wd = self.config.transfer_config.finetune_wd
      wd_losses = list(
          map(lambda x: 0.5 * wd * tf.reduce_sum(tf.square(x)), var_list))
      log.info('Fast weight decay variables')
      [log.info(x) for x in wd_losses]
      return tf.add_n(wd_losses)
    else:
      log.warning('No fast weight decay variables!')
      return 0.0

  def get_slow_weights(self):
    """Returns a set of slow weights."""
    var_list = tf.trainable_variables()
    var_list = list(filter(lambda x: 'phi' in x.name, var_list))
    layers = self.config.transfer_config.meta_layers
    if layers == "all":
      pass
    elif layers == "4":
      keywords = ['TaskB', 'unit_4_']
      filter_fn = lambda x: any([kw in x.name for kw in keywords])
      var_list = list(filter(filter_fn, var_list))
    else:
      raise ValueError('Unknown finetune layers {}'.format(layers))
    [log.info('Slow weights {}'.format(v.name)) for v in var_list]
    return var_list

  def get_transfer_loss_weights(self, name='transfer_loss'):
    var_list = tf.trainable_variables()
    var_list = list(filter(lambda x: name in x.name, var_list))
    return var_list

  def get_meta_weights(self):
    """Returns a set of weights that belongs to the meta-learner."""
    var_list = self.get_transfer_loss_weights(name=self.transfer_loss_name)
    var_list += self.get_transfer_loss_weights(name=self.new_loss_name)
    proto_config = self.config.protonet_config
    transfer_config = self.config.transfer_config
    if proto_config.cosine_softmax_tau:
      var_list += [self._tau_b]

    if proto_config.protos_phi:
      var_list += [self._w_p1]

    if transfer_config.train_wclass_a:
      var_list += [self.w_class_a]
      if not proto_config.cosine_softmax:
        var_list += [self.b_class_a]
    return var_list

  def get_optimizer(self, optname, learn_rate):
    """Gets an optimizer."""
    if optname == 'adam':
      opt = tf.train.AdamOptimizer(learn_rate)
    elif optname == 'momentum':
      opt = tf.train.MomentumOptimizer(learn_rate, 0.9)
    elif optname == 'nesterov':
      opt = tf.train.MomentumOptimizer(learn_rate, 0.9, use_nesterov=True)
    elif optname == 'sgd':
      opt = tf.train.GradientDescentOptimizer(learn_rate)
    else:
      raise ValueError('Unknown optimizer')
    return opt

  def get_fdict(self, task_a_data=None, task_b_data=None):
    """Make a feed dict."""
    fdict = {}

    if task_a_data is not None:
      x_a, y_a = task_a_data
      fdict[self.inputs] = x_a
      fdict[self.labels] = y_a

    if task_b_data is not None:
      fdict[self.inputs_b] = task_b_data.x_train
      fdict[self.labels_b] = task_b_data.y_train
      fdict[self.inputs_b_v] = task_b_data.x_test
      fdict[self.labels_b_v] = task_b_data.y_test

      if task_b_data.y_sel is not None:
        fdict[self._y_sel] = task_b_data.y_sel
        fdict[self._mask] = True
        # print('adding ysel', task_b_data.y_sel)
      else:
        fdict[self._y_sel] = np.zeros([self._num_classes_b], dtype=np.int64)
        fdict[self._mask] = False
        # print('not adding ysel')
    return fdict

  def train_step_a(self, sess, task_a_data):
    """Train a single step on task A."""
    x_a, y_a = task_a_data
    fdict = self.get_fdict(task_a_data=task_a_data)
    cost_a, _ = sess.run([self.cost_a, self.train_op_a], feed_dict=fdict)
    return cost_a

  def eval_step(self, sess, task_a_data, task_b_data):
    """Evaluate one step."""
    prediction_a, y_a = self.eval_step_a(sess, task_a_data)
    prediction_b, y_b = self.eval_step_b(sess, task_b_data)
    return prediction_a, prediction_b

  def eval_step_a(self, sess, task_a_data):
    """Evaluate one step on task A."""
    x_a, y_a = task_a_data
    fdict = self.get_fdict(task_a_data=task_a_data)
    prediction_a, y_a = sess.run([self.prediction_a_all, self.labels_all],
                                 feed_dict=fdict)
    return prediction_a, y_a

  def eval_step_b(self, sess, task_b_data):
    """Evaluate one step on task B."""
    raise NotImplemented()

  def eval_step_b_old_and_new(self, sess, task_b_data):
    """Evaluate one step when there is both old and new data."""
    raise NotImplemented()

  def train_step(self, sess, task_a_data):
    """Train a single step."""
    raise NotImplemented()

  @property
  def transfer_loss_name(self):
    return "transfer_loss"

  @property
  def new_loss_name(self):
    return "new_loss"

  @property
  def global_step(self):
    return tf.contrib.framework.get_or_create_global_step()

  @property
  def inputs(self):
    """Input images on task A."""
    return self._inputs

  @property
  def labels(self):
    """Labels on task A."""
    return self._labels

  @property
  def labels_all(self):
    """All labels on task A."""
    return self._labels_all

  @property
  def inputs_b(self):
    """Input images on task B."""
    return self._inputs_b

  @property
  def labels_b(self):
    """Labels on task B."""
    return self._labels_b

  @property
  def inputs_b_v(self):
    """Input images on task B query."""
    return self._inputs_b_v

  @property
  def labels_b_v(self):
    """All labels on task B support."""
    return self._labels_b_v

  @property
  def labels_b_v_all(self):
    """All labels on task B query."""
    return self._labels_b_v_all

  @property
  def cost_a(self):
    """Loss on task A."""
    return self._cost_a

  @property
  def cost_b(self):
    """Loss on task B support."""
    return self._cost_b

  @property
  def cost_b_v(self):
    """Loss on task B query."""
    return self._cost_b_v

  @property
  def acc_a(self):
    """Accuracy on task A."""
    return self._acc_a

  @property
  def acc_b(self):
    """Accuracy on task B."""
    return self._acc_b

  @property
  def w_class_a(self):
    """Weights for task A classifier."""
    return self._w_class_a

  @property
  def b_class_a(self):
    """Bias for task A classifier."""
    return self._b_class_a

  @property
  def h_a(self):
    """Hidden state for task A."""
    return self._h_a

  @property
  def prediction_a(self):
    """Prediction on task A."""
    return self._prediction_a

  @property
  def prediction_a_all(self):
    """All prediction on task A."""
    return self._prediction_a_all

  @property
  def dtype(self):
    """Data type."""
    if self.config.dtype == 'float32':
      return tf.float32
    elif self.config.dtype == 'float64':
      return tf.float64

  @property
  def attractor(self):
    """Attractor module."""
    return self._attractor
