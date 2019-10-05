"""Imprint model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

from horovod.tensorflow.mpi_ops import allgather

from fewshot.models.kmeans_utils import compute_logits_cosine
from fewshot.models.model_factory import get_model
from fewshot.models.nnlib import weight_variable
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ImprintModel(object):
  """A model with both regular classification branch and few-shot branch."""

  def __init__(self,
               config,
               x,
               y,
               x_b,
               y_b,
               x_b_v,
               y_b_v,
               num_classes_a,
               num_classes_b,
               is_training=True,
               ext_wts=None,
               y_sel=None,
               w_class_a=None,
               b_class_a=None):
    self._config = config
    self._is_training = is_training
    self._num_classes_a = num_classes_a
    self._num_classes_b = num_classes_b

    if config.backbone_class == 'resnet_backbone':
      bb_config = config.resnet_config
    else:
      assert False, 'Not supported'
    opt_config = config.optimizer_config
    proto_config = config.protonet_config
    transfer_config = config.transfer_config

    self._backbone = get_model(config.backbone_class, bb_config)
    self._inputs = x
    self._labels = y
    if opt_config.num_gpu > 1:
      self._labels_all = allgather(self._labels)
    else:
      self._labels_all = self._labels
    self._inputs_b = x_b
    self._labels_b = y_b
    self._inputs_b_v = x_b_v
    self._labels_b_v = y_b_v
    if opt_config.num_gpu > 1:
      self._labels_b_v_all = allgather(self._labels_b_v)
    else:
      self._labels_b_v_all = self._labels_b_v
    self._y_sel = y_sel
    self._mask = tf.placeholder(tf.bool, [], name='mask')

    # global_step = tf.get_variable(
    #     'global_step', shape=[], dtype=tf.int64, trainable=False)
    global_step = tf.contrib.framework.get_or_create_global_step()
    self._global_step = global_step
    log.info('LR decay steps {}'.format(opt_config.lr_decay_steps))
    log.info('LR list {}'.format(opt_config.lr_list))
    learn_rate = tf.train.piecewise_constant(
        global_step, list(
            np.array(opt_config.lr_decay_steps).astype(np.int64)),
        list(opt_config.lr_list))
    self._learn_rate = learn_rate

    opt = self.get_optimizer(opt_config.optimizer, learn_rate)
    if opt_config.num_gpu > 1:
      opt = hvd.DistributedOptimizer(opt)

    with tf.name_scope('TaskA'):
      h_a = self.backbone(x, is_training=is_training, ext_wts=ext_wts)
      self._h_a = h_a

    # Apply BN ops.
    bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope('TaskB'):
      x_b_all = tf.concat([x_b, x_b_v], axis=0)
      if ext_wts is not None:
        h_b_all = self.backbone(
            x_b_all, is_training=is_training, reuse=True, ext_wts=ext_wts)
      else:
        h_b_all = self.backbone(x_b_all, is_training=is_training, reuse=True)

    with tf.name_scope('TaskA'):
      # Calculates hidden activation size.
      h_shape = h_a.get_shape()
      h_size = 1
      for ss in h_shape[1:]:
        h_size *= int(ss)

      if w_class_a is None:
        if ext_wts is not None:
          w_class_a = weight_variable(
              [h_size, num_classes_a],
              init_method='numpy',
              dtype=tf.float32,
              init_param={'val': np.transpose(ext_wts['w_class_a'])},
              wd=config.wd,
              name='w_class_a')
          b_class_a = weight_variable([],
                                      init_method='numpy',
                                      dtype=tf.float32,
                                      init_param={'val': ext_wts['b_class_a']},
                                      wd=0e0,
                                      name='b_class_a')
        else:
          w_class_a = weight_variable([h_size, num_classes_a],
                                      init_method='truncated_normal',
                                      dtype=tf.float32,
                                      init_param={'stddev': 0.01},
                                      wd=bb_config.wd,
                                      name='w_class_a')
          b_class_a = weight_variable([num_classes_a],
                                      init_method='constant',
                                      init_param={'val': 0.0},
                                      name='b_class_a')
        self._w_class_a_orig = w_class_a
        self._b_class_a_orig = b_class_a
      else:
        assert b_class_a is not None
        w_class_a_orig = weight_variable([h_size, num_classes_a],
                                         init_method='truncated_normal',
                                         dtype=tf.float32,
                                         init_param={'stddev': 0.01},
                                         wd=bb_config.wd,
                                         name='w_class_a')
        b_class_a_orig = weight_variable([num_classes_a],
                                         init_method='constant',
                                         init_param={'val': 0.0},
                                         name='b_class_a')
        self._w_class_a_orig = w_class_a_orig
        self._b_class_a_orig = b_class_a_orig

      self._w_class_a = w_class_a
      self._b_class_a = b_class_a
      num_classes_a_dyn = tf.cast(tf.shape(b_class_a)[0], tf.int64)
      num_classes_a_dyn32 = tf.shape(b_class_a)[0]

      if ext_wts is None:
        init_val = 10.0
      else:
        init_val = ext_wts['tau'][0]
      tau = weight_variable([],
                            init_method='constant',
                            init_param={'val': init_val},
                            name='tau')
      w_class_a_norm = self._normalize(w_class_a, 0)
      h_a_norm = self._normalize(h_a, 1)
      dot = tf.matmul(h_a_norm, w_class_a_norm)
      if ext_wts is not None:
        dot += b_class_a
      logits_a = tau * dot
      self._prediction_a = logits_a
      if opt_config.num_gpu > 1:
        self._prediction_a_all = allgather(self._prediction_a)
      else:
        self._prediction_a_all = self._prediction_a

      xent_a = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_a, labels=y)
      cost_a = tf.reduce_mean(xent_a, name='xent')
      self._cost_a = cost_a
      cost_a += self._decay()
      correct_a = tf.equal(tf.argmax(logits_a, axis=1), y)
      self._correct_a = correct_a
      self._acc_a = tf.reduce_mean(tf.cast(correct_a, cost_a.dtype))

    with tf.name_scope('TaskB'):
      h_b = h_b_all[:tf.shape(x_b)[0]]
      h_b_v = h_b_all[tf.shape(x_b)[0]:]

      # Add new axes for the `batch` dimension.
      h_b_ = tf.expand_dims(h_b, 0)
      h_b_v_ = tf.expand_dims(h_b_v, 0)
      y_b_ = tf.expand_dims(y_b, 0)
      y_b_v_ = tf.expand_dims(y_b_v, 0)
      protos_b = self._compute_protos(num_classes_b, h_b_,
                                      y_b_ - num_classes_a)
      w_class_a_ = tf.expand_dims(tf.transpose(w_class_a_norm), 0)  # [1, K, D]
      w_class_b = self._normalize(protos_b, 2)  # [1, K, D]
      self._w_class_b = w_class_b
      w_class_all = tf.concat([w_class_a_, w_class_b], axis=1)
      logits_b_v = tau * compute_logits_cosine(w_class_all, h_b_v_)
      self._logits_b_v = logits_b_v
      self._prediction_b = logits_b_v[0]
      if opt_config.num_gpu > 1:
        self._prediction_b_all = allgather(self._prediction_b)
      else:
        self._prediction_b_all = self._prediction_b

      # Mask out the old classes.
      def mask_fn():
        bin_mask = tf.expand_dims(
            tf.reduce_sum(
                tf.one_hot(y_sel, num_classes_a + num_classes_b),
                0,
                keep_dims=True), 0)
        logits_b_v_m = logits_b_v * (1.0 - bin_mask)
        logits_b_v_m -= bin_mask * 100.0
        return logits_b_v_m

      if transfer_config.old_and_new:
        logits_b_v = tf.cond(self._mask, mask_fn, lambda: logits_b_v)
      xent_b_v = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_b_v, labels=y_b_v_)
      cost_b = tf.reduce_mean(xent_b_v, name='xent')
      self._cost_b = cost_b

    if transfer_config.old_and_new:
      total_cost = cost_b
    else:
      total_cost = (transfer_config.cost_a_ratio * cost_a +
                    transfer_config.cost_b_ratio * cost_b)
    self._total_cost = total_cost

    if not transfer_config.meta_only:
      # assert False, 'let us go for pretrained model first'
      var_list = tf.trainable_variables()
      var_list = list(filter(lambda x: 'phi' in x.name, var_list))
      [log.info('Slow weights {}'.format(v.name)) for v in var_list]
    else:
      var_list = []

    if is_training:
      grads_and_vars = opt.compute_gradients(total_cost, var_list)
      with tf.control_dependencies(bn_ops):
        [log.info('BN op {}'.format(op.name)) for op in bn_ops]
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

      grads_and_vars_b = opt.compute_gradients(cost_b, var_list)
      with tf.control_dependencies(bn_ops):
        train_op_b = opt.apply_gradients(
            grads_and_vars_b, global_step=global_step)

      with tf.control_dependencies(bn_ops):
        train_op_a = opt.minimize(cost_a, global_step=global_step)
      self._train_op = train_op
      self._train_op_a = train_op_a
      self._train_op_b = train_op_b
    self._initializer = tf.global_variables_initializer()

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
      # [B, N, 1]
      ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
      # [B, N, D]
      protos[kk] = tf.reduce_sum(h_train * ksel, [1], keep_dims=True)
      protos[kk] /= tf.reduce_sum(ksel, [1, 2], keep_dims=True)
    protos = tf.concat(protos, axis=1)  # [B, K, D]
    return protos

  def _decay(self):
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info('Weight decay variables')
    [log.info(x) for x in wd_losses]
    log.info('Total length: {}'.format(len(wd_losses)))
    if len(wd_losses) > 0:
      return tf.add_n(wd_losses)
    else:
      log.warning('No weight decay variables!')
      return 0.0

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
      else:
        fdict[self._y_sel] = np.zeros([self._num_classes_b], dtype=np.int64)
        fdict[self._mask] = False
    return fdict

  def get_optimizer(self, optname, learn_rate):
    """Gets an optimizer.

    Args:
      optname: String. Name of the optimizer.
    """
    if optname == 'adam':
      opt = tf.train.AdamOptimizer(learn_rate)
    elif optname == 'momentum':
      opt = tf.train.MomentumOptimizer(learn_rate, 0.9)
    elif optname == 'nesterov':
      opt = tf.train.MomentumOptimizer(learn_rate, 0.9, use_nesterov=True)
    else:
      raise ValueError('Unknown optimizer {}'.format(optname))
    return opt

  def initialize(self, sess):
    """Initialize model."""
    sess.run(self._initializer)

  def eval_step_a(self, sess, task_a_data):
    """Evaluate one step on task A."""
    x_a, y_a = task_a_data
    fdict = {self.inputs: x_a, self.labels: y_a}
    prediction_a, y_a = sess.run([self.prediction_a_all, self.labels_all],
                                 feed_dict=fdict)
    return prediction_a, y_a

  def eval_step_b(self, sess, task_b_data):
    """Evaluate one step on task B."""
    prediction_b, y_b = sess.run(
        [self.prediction_b_all, self.labels_b_v_all],
        feed_dict={
            self.inputs_b: task_b_data.x_train,
            self.labels_b: task_b_data.y_train,
            self.inputs_b_v: task_b_data.x_test,
            self.labels_b_v: task_b_data.y_test
        })
    return prediction_b, y_b

  def eval_step(self, sess, task_a_data, task_b_data):
    """Evaluate one step."""
    prediction_a = self.eval_step_a(sess, task_a_data)
    prediction_b = self.eval_step_b(sess, task_b_data)
    return prediction_a, prediction_b

  def train_step(self, sess, task_a_data, task_b_data):
    """Train a single step, for optimizing a combined loss."""
    fdict = self.get_fdict(task_a_data, task_b_data)
    cost_a, cost_b, total_cost, _ = sess.run(
        [self.cost_a, self.cost_b, self.total_cost, self.train_op],
        feed_dict=fdict)
    return cost_a, 0.0, cost_b

  def train_step_a(self, sess, task_a_data):
    """Train a single step on task A."""
    x_a, y_a = task_a_data
    fdict = {self.inputs: x_a, self.labels: y_a}
    cost_a, _ = sess.run([self.cost_a, self.train_op_a], feed_dict=fdict)
    return cost_a

  def train_step_b(self, sess, task_b_data):
    """Train a single step on task B."""
    fdict = {
        self.inputs_b: task_b_data.x_train,
        self.labels_b: task_b_data.y_train,
        self.inputs_b_v: task_b_data.x_test,
        self.labels_b_v: task_b_data.y_test
    }
    if task_b_data.y_sel is not None:
      fdict[self._y_sel] = task_b_data.y_sel
      fdict[self._mask] = True
    else:
      fdict[self._y_sel] = np.zeros([self._num_classes_b], dtype=np.int64)
      fdict[self._mask] = False
    cost_b, _ = sess.run([self.cost_b, self.train_op_b], feed_dict=fdict)
    return cost_b

  def _get_optimizer(self, optname, learn_rate):
    """Gets an optimizer."""
    if optname == 'adam':
      opt = tf.train.AdamOptimizer(learn_rate)
    elif optname == 'momentum':
      opt = tf.train.MomentumOptimizer(learn_rate, 0.9)
    elif optname == 'nesterov':
      opt = tf.train.MomentumOptimizer(learn_rate, 0.9, use_nesterov=True)
    else:
      raise ValueError('Unknown optimizer')
    return opt

  def _normalize(self, x, axis, eps=1e-5):
    return x / (
        tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)) + 1e-5)

  @property
  def backbone(self):
    return self._backbone

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def labels_all(self):
    return self._labels_all

  @property
  def inputs_b(self):
    return self._inputs_b

  @property
  def labels_b(self):
    return self._labels_b

  @property
  def inputs_b_v(self):
    return self._inputs_b_v

  @property
  def labels_b_v(self):
    return self._labels_b_v

  @property
  def labels_b_v_all(self):
    return self._labels_b_v_all

  @property
  def cost_a(self):
    return self._cost_a

  @property
  def cost_b(self):
    return self._cost_b

  @property
  def total_cost(self):
    return self._total_cost

  @property
  def train_op(self):
    return self._train_op

  @property
  def config(self):
    return self._config

  @property
  def learn_rate(self):
    return self._learn_rate

  @property
  def train_op_a(self):
    return self._train_op_a

  @property
  def train_op_b(self):
    return self._train_op_b

  @property
  def h_a(self):
    return self._h_a

  @property
  def prediction_a(self):
    return self._prediction_a

  @property
  def prediction_a_all(self):
    return self._prediction_a_all

  @property
  def prediction_b(self):
    return self._prediction_b

  @property
  def prediction_b_all(self):
    return self._prediction_b_all

  @property
  def num_classes_a(self):
    return self._num_classes_a

  @property
  def num_classes_b(self):
    return self._num_classes_b
