"""Attractor model.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six
import tensorflow as tf

from tqdm import tqdm

from fewshot.models.model_factory import get_model
from fewshot.models.nnlib import weight_variable
from fewshot.models.rbp import rbp
from fewshot.models.resnet_backbone import ResnetBackbone  # NOQA
from fewshot.models.basic_backbone import BasicBackbone  # NOQA
from fewshot.models.fc_backbone import FCBackbone  # NOQA
from fewshot.models.attractor_model_base import AttractorModelBase
from fewshot.utils.checkpoint import build_checkpoint
from fewshot.utils.checkpoint import write_checkpoint
from fewshot.utils.logger import get as get_logger

log = get_logger()


class AttractorModel(AttractorModelBase):
  """Attractor model."""

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
               y_sel=None,
               ext_wts=None):
    """Attractor model with RBP.

    Args:
      config: Model config object.
      x: Inputs on task A.
      y: Labels on task A.
      x_b: Support inputs on task B.
      y_b: Support labels on task B.
      x_b_v: Query inputs on task B.
      y_b_v: Query labels on task B.
      num_classes_a: Number of classes on task A.
      num_classes_b: Number of classes on task B.
      is_training: Whether in training mode.
      y_sel: Mask on base classes.
      ext_wts: External weights for initialization.
    """
    self._config = config
    self._is_training = is_training
    self._num_classes_a = num_classes_a
    self._num_classes_b = num_classes_b
    self._global_step = None

    if config.backbone_class == 'resnet_backbone':
      bb_config = config.resnet_config
    else:
      assert False, 'Not supported'
    opt_config = config.optimizer_config
    proto_config = config.protonet_config
    transfer_config = config.transfer_config
    ft_opt_config = transfer_config.ft_optimizer_config

    self._backbone = get_model(config.backbone_class, bb_config)
    self._inputs = x
    self._labels = y
    self._labels_all = self._labels

    self._y_sel = y_sel
    self._rnd = np.random.RandomState(0)  # Common random seed.

    # A step counter for the meta training stage.
    global_step = self.global_step

    log.info('LR decay steps {}'.format(opt_config.lr_decay_steps))
    log.info('LR list {}'.format(opt_config.lr_list))

    # Learning rate decay.
    learn_rate = tf.train.piecewise_constant(
        global_step, list(
            np.array(opt_config.lr_decay_steps).astype(np.int64)),
        list(opt_config.lr_list))
    self._learn_rate = learn_rate

    # Class matrix mask.
    self._mask = tf.placeholder(tf.bool, [], name='mask')

    # Optimizer definition.
    opt = self.get_optimizer(opt_config.optimizer, learn_rate)

    # Task A branch.
    with tf.name_scope('TaskA'):
      self.build_task_a(x, y, is_training, ext_wts=ext_wts)
      if is_training:
        grads_and_vars_a = self.build_task_a_grad()
        with tf.variable_scope('Optimizer'):
          bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          with tf.control_dependencies(bn_ops):
            self._train_op_a = opt.apply_gradients(
                grads_and_vars_a, global_step=global_step)
    h_size = self._h_size  # Calculated in the function above.
    w_class_a = self.w_class_a
    b_class_a = self.b_class_a

    # The finetuning task.
    self._inputs_b = x_b
    self._labels_b = y_b
    self._inputs_b_v = x_b_v
    self._labels_b_v = y_b_v
    self._labels_b_v_all = y_b_v

    with tf.name_scope('TaskB'):
      self.build_task_b(x_b, y_b, x_b_v, y_sel)
      if is_training:
        grads_and_vars_b = self.build_task_b_grad(x_b_v, y_b_v, y_sel)

    # Task A and Task B cost weights.
    assert transfer_config.cost_a_ratio == 0.0
    assert transfer_config.cost_b_ratio == 1.0
    cost_a_ratio_var = tf.constant(
        transfer_config.cost_a_ratio, name='cost_a_ratio', dtype=self.dtype)
    cost_b_ratio_var = tf.constant(
        transfer_config.cost_b_ratio, name='cost_b_ratio', dtype=self.dtype)

    # Update gradients for meta-leraning.
    if is_training:
      total_grads_and_vars_ab = self._aggregate_grads_and_vars(
          [grads_and_vars_a, grads_and_vars_b],
          weights=[cost_a_ratio_var, cost_b_ratio_var])
      with tf.variable_scope('Optimizer'):
        with tf.control_dependencies(bn_ops):
          self._train_op = opt.apply_gradients(
              total_grads_and_vars_ab, global_step=global_step)

      if len(grads_and_vars_b) > 0:
        self._train_op_b = opt.apply_gradients(
            grads_and_vars_b, global_step=global_step)
      else:
        self._train_op_b = tf.no_op()

    self._initializer = tf.global_variables_initializer()

  def build_fast_weights(self):
    """Build fast weights for task B."""
    transfer_config = self.config.transfer_config
    num_classes_b = self.num_classes_b
    h_size = self._h_size
    suffix = '' if self.num_classes_b == 5 else '_{}'.format(num_classes_b)
    if transfer_config.fast_model == 'lr':
      w_class_b = weight_variable([h_size, num_classes_b],
                                  dtype=self.dtype,
                                  init_method='constant',
                                  init_param={'val': 0.0},
                                  wd=transfer_config.finetune_wd,
                                  name='w_class_b' + suffix)
      b_class_b = weight_variable([num_classes_b],
                                  dtype=self.dtype,
                                  init_method='constant',
                                  init_param={'val': -1.0},
                                  name='b_class_b' + suffix)
      fast_weights = [w_class_b, b_class_b]
    elif transfer_config.fast_model == 'resmlp':
      mlp_size = transfer_config.fast_mlp_hidden
      w_class_b2 = weight_variable([h_size, mlp_size],
                                   dtype=self.dtype,
                                   init_method='constant',
                                   init_param={'val': 0.0},
                                   wd=transfer_config.finetune_wd,
                                   name='w_class_b2' + suffix)
      b_class_b2 = weight_variable([mlp_size],
                                   dtype=self.dtype,
                                   init_method='constant',
                                   init_param={'val': 0.0},
                                   name='b_class_b2' + suffix)
      w_class_b3 = weight_variable([h_size, num_classes_b],
                                   dtype=self.dtype,
                                   init_method='constant',
                                   init_param={'val': 0.0},
                                   wd=transfer_config.finetune_wd,
                                   name='w_class_b3' + suffix)
      w_class_b = weight_variable([mlp_size, num_classes_b],
                                  dtype=self.dtype,
                                  init_method='constant',
                                  init_param={'val': 0.0},
                                  wd=transfer_config.finetune_wd,
                                  name='w_class_b' + suffix)
      b_class_b = weight_variable([num_classes_b],
                                  dtype=self.dtype,
                                  init_method='constant',
                                  init_param={'val': -1.0},
                                  name='b_class_b' + suffix)
      fast_weights = [w_class_b3, w_class_b2, b_class_b2, w_class_b, b_class_b]
    else:
      assert False
    return fast_weights

  def build_task_b_fast_optimizer(self, cost, fast_weights):
    """Build task B optimizer here."""
    transfer_config = self.config.transfer_config
    ft_opt_config = transfer_config.ft_optimizer_config
    if ft_opt_config.optimizer not in ['lbfgs']:
      grads_fast = tf.gradients(cost, fast_weights, gate_gradients=1)
      grads_and_vars_fast = zip(grads_fast, fast_weights)
      # A step counter for the finetuning stage.
      ft_step = tf.get_variable(
          'ft_step',
          shape=[],
          dtype=tf.int64,
          trainable=False,
          initializer=tf.constant_initializer(0))
      self._ft_step = ft_step
      # Learning rate for finetuning.
      ft_learn_rate = tf.train.piecewise_constant(
          ft_step, list(
              np.array(ft_opt_config.lr_decay_steps).astype(np.int64)),
          list(ft_opt_config.lr_list))
      ftopt = self.get_optimizer(ft_opt_config.optimizer, ft_learn_rate)
      self._train_op_ft = ftopt.apply_gradients(
          grads_and_vars_fast, global_step=ft_step)

    elif ft_opt_config.optimizer in ['lbfgs']:
      disp = False
      scipy_interface = transfer_config.scipy_interface
      if scipy_interface == 'built-in':
        log.info('Using built-in scipy optimizer interface.')
        assert ft_opt_config.batch_size == -1
        interface = tf.contrib.opt.ScipyOptimizerInterface
      else:
        assert False, 'Unknown scipy interface'
      ftopt_scipy = interface(
          cost,
          var_list=fast_weights,
          method='L-BFGS-B',
          options={
              'disp': disp,
              'maxiter': 10000
          })
      self._ft_opt_scipy = ftopt_scipy
    else:
      self._ft_opt_scipy = None

  def build_task_b(self, x_b, y_b, x_b_v, y_sel):
    """Build task B.

    Args:
      x_b: Tensor. [S, H, W, C]. Support tensor.
      y_b: Tensor. [S]. Support labels.
      x_b_v: Tensor. [Q, H, W, C]. Query tensor.
      y_sel: Tensor. [K]. Mask class tensor.
    """
    transfer_config = self.config.transfer_config
    proto_config = self.config.protonet_config
    ft_opt_config = transfer_config.ft_optimizer_config
    opt_config = self.config.optimizer_config
    is_training = self._is_training
    h_size = self._h_size
    num_classes_a = self._num_classes_a
    num_classes_b = self._num_classes_b
    w_class_a = self._w_class_a
    b_class_a = self._b_class_a
    y_sel = self._y_sel
    old_and_new = transfer_config.old_and_new
    assert not proto_config.cosine_softmax_tau

    # Build fast classifier weights.
    fast_weights = self.build_fast_weights()

    # Finetune weights use placeholder to initialize.
    init_ops = []
    for v in fast_weights:
      init_ops.append(
          tf.assign(
              v,
              tf.constant(
                  self._rnd.uniform(-0.01, 0.01, [int(ss) for ss in v.shape]),
                  dtype=v.dtype)))
    self._init_ops = init_ops

    h_b = self._run_backbone(x_b, reuse=True, is_training=is_training)
    h_b_plh = tf.placeholder(self.dtype, [None, h_size], name='h_b_plh')
    self._hidden_b = h_b
    self._hidden_b_plh = h_b_plh

    # Use checkpointed hidden.
    assert transfer_config.finetune_layers == 'none'
    if self.save_hidden_b:
      h_b = h_b_plh
    self._h_b = h_b

    # Joint prediction.
    w_class_a_mask = tf.cond(self._mask, self.get_mask_fn_wa(w_class_a, y_sel),
                             lambda: w_class_a)
    self._w_class_a_mask = w_class_a_mask

    # Whether learning on a combination of old and new or just new.
    if old_and_new:
      y_b_dense = tf.one_hot(y_b, num_classes_a + num_classes_b)
    else:
      y_b_dense = tf.one_hot(y_b, num_classes_b)

    # Unroll forward graph.
    # fast_weights0 = [w for w in fast_weights]
    fast_weights0 = fast_weights
    self.fast_weights0 = fast_weights0
    dummy_lr = transfer_config.dummy_lr
    for step in range(transfer_config.bptt_steps):
      logits_b = self.compute_logits_b_all(h_b, fast_weights, w_class_a_mask,
                                           b_class_a)
      xent_b = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits_b, labels=y_b_dense)
      cost_b = tf.reduce_mean(xent_b)
      ft_decay = self._ft_decay(fast_weights)
      cost_b += ft_decay
      # Add another transfer loss to prevent catastrophic forgetting.
      cost_b += self._apply_transfer_loss(
          fast_weights,
          h_b=h_b,
          y_b=y_b,
          w_class_a=w_class_a_mask,
          b_class_a=b_class_a,
          reuse=True if step > 0 else None)
      fast_grads = tf.gradients(cost_b, fast_weights, gate_gradients=1)
      if step == 0:
        # Use the first step to build gradients for external optimizer.
        self.build_task_b_fast_optimizer(cost_b, fast_weights)

      # Dummy gradient descent.
      fast_weights = [
          vv - dummy_lr * gg if gg is not None else vv
          for gg, vv in zip(fast_grads, fast_weights)
      ]
    fast_weights1 = fast_weights
    self.fast_weights1 = fast_weights1
    self._cost_b = cost_b

    # Training accuracy.
    if old_and_new:
      y_b_tr = y_b - num_classes_a
      pred_b_tr = tf.argmax(logits_b[:, num_classes_a:], axis=-1)
    else:
      y_b_tr = y_b
      pred_b_tr = tf.argmax(logits_b, axis=-1)
    correct_b_tr = tf.equal(pred_b_tr, y_b_tr)
    self._acc_b_tr = tf.reduce_mean(tf.cast(correct_b_tr, cost_b.dtype))

    # Evaluate accuracy, not on T+1, to remove dependency from task B training
    # data.
    h_b_v2 = self._run_backbone(x_b_v, reuse=True, is_training=is_training)
    logits_b_v2 = self.compute_logits_b_all(h_b_v2, fast_weights,
                                            w_class_a_mask, b_class_a)
    self._prediction_b = logits_b_v2
    self._prediction_b_all = self._prediction_b
    correct_b_v = tf.equal(
        tf.argmax(self._prediction_b_all, axis=-1), self._labels_b_v_all)
    self._acc_b_v = tf.reduce_mean(tf.cast(correct_b_v, cost_b.dtype))
    return logits_b_v2

  # Mask out the old classes.
  def get_mask_fn(self, logits, y_sel):
    """Mask the logits."""
    num_classes_a = self.num_classes_a
    num_classes_b = self.num_classes_b

    def _mask_fn():
      y_dense = tf.one_hot(
          self._y_sel, num_classes_a + num_classes_b, dtype=self.dtype)
      bin_mask = tf.reduce_sum(y_dense, 0, keep_dims=True)
      logits_mask = logits * (1.0 - bin_mask)
      logits_mask -= bin_mask * 100.0
      return logits_mask

    return _mask_fn

  def resmlp(self, x, w3, w2, b2, w, b):
    """MLP with a residual connection."""
    return tf.matmul(tf.nn.tanh(tf.matmul(x, w2) + b2), w) + tf.matmul(x,
                                                                       w3) + b

  def fc(self, x, w, b):
    """Fully connected layer."""
    return tf.matmul(x, w) + b

  # Mask out the old classes.
  def get_mask_fn_wa(self, w_class_a, y_sel):
    """Mask the weights in task A."""
    num_classes_a = self.num_classes_a

    def _mask_fn():
      y_dense = tf.one_hot(y_sel, num_classes_a, dtype=self.dtype)
      bin_mask = tf.reduce_sum(y_dense, 0, keep_dims=True)
      wa = w_class_a * (1.0 - bin_mask) + 1e-7 * bin_mask
      return wa

    return _mask_fn

  def compute_logits_b(self, h, weights):
    """Compute logits for task B branch."""
    transfer_config = self.config.transfer_config
    if transfer_config.fast_model == 'lr':
      logits_b_v = self.fc(h, *weights)
    elif transfer_config.fast_model == 'resmlp':
      logits_b_v = self.resmlp(h, *weights)
    return logits_b_v

  def compute_logits_b_all(self, h, w_b_list, w_a, b_a):
    """Compute logits for task B branch, possibly combined A logits."""
    logits_b = self.compute_logits_b(h, w_b_list)
    y_sel = self._y_sel
    if self.config.transfer_config.old_and_new:
      logits_a = self.fc(h, w_a, b_a)
      logits_b = tf.concat([logits_a, logits_b], 1)
      logits_b = tf.cond(self._mask, self.get_mask_fn(logits_b, y_sel),
                         lambda: logits_b)
    return logits_b

  def build_task_b_grad(self, x_b_v, y_b_v, y_sel):
    """Build gradients for task B.

    Args:
      x_b_v: Tensor. [Q, H, W, C]. Query tensor.
      y_b_v: Tensor. [Q]. Query label.
    """
    config = self.config
    transfer_config = config.transfer_config

    # This is the meta-learning gradients.
    assert transfer_config.meta_only
    slow_weights = []

    meta_weights = self.get_meta_weights()
    num_classes_a = self.num_classes_a
    num_classes_b = self.num_classes_b
    w_class_a = self.w_class_a
    w_class_a_mask = self._w_class_a_mask
    b_class_a = self.b_class_a
    old_and_new = transfer_config.old_and_new
    is_training = self._is_training
    fast_weights0 = self.fast_weights0
    fast_weights1 = self.fast_weights1
    debug_rbp = False

    # Run again on the validation.
    h_b_v = self._run_backbone(x_b_v, reuse=True, is_training=is_training)

    # Loss on the task B validation set, for meta-learning.
    logits_b_v = self.compute_logits_b_all(h_b_v, fast_weights1,
                                           w_class_a_mask, b_class_a)
    xent_b_v = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_b_v, labels=y_b_v))
    self._cost_b_v = xent_b_v
    meta_cost = xent_b_v

    # Stores gradients.
    grads_b_ckpt = build_checkpoint(slow_weights + meta_weights, 'grads_b')
    transfer_config = self.config.transfer_config
    assert transfer_config.finetune_layers == 'none'
    has_tloss_cache = transfer_config.cache_transfer_loss_var
    assert not has_tloss_cache

    # Target of gradients.
    x = meta_weights  # [Trans + Meta]

    # Run RBP algorithm to get gradients.
    grads_b_rbp = rbp(
        x,
        fast_weights1,
        fast_weights0,
        meta_cost,
        transfer_config.rbp_nstep,
        lbd=transfer_config.rbp_lambda,
        debug=debug_rbp)
    grads_slow = []

    # Write to gradient variables.
    log.info('Meta grads')
    grads_meta = grads_b_rbp

    [
        log.info(x.name + ': ' + grad.name)
        for x, grad in zip(meta_weights, grads_meta)
    ]
    self._meta_weights = meta_weights
    self._grads_meta = grads_meta
    update_grads_b = write_checkpoint(grads_b_ckpt, grads_slow + grads_meta)
    self._update_grads_b = update_grads_b
    return list(zip(grads_b_ckpt, slow_weights + meta_weights))

  def minibatch(self, x, y, batch_size, rnd=None, num=0):
    """Samples a mini-batch from the episode."""
    if batch_size == -1:
      """Full batch mode"""
      return x, y
    idx = np.arange(x.shape[0])
    if rnd is not None:
      rnd.shuffle(idx)
      idx = idx[:batch_size]
      return x[idx], y[idx]
    else:
      length = x.shape[0]
      start = num * batch_size
      end = min((num + 1) * batch_size, length)
      return x[start:end], y[start:end]

  def monitor_b(self, sess, x_b_np, y_b_np, x_b_v_np, y_b_v_np, fdict=None):
    """Solve the few-shot classifier and monitor the progress on training and
    validation sets.

    Args:
      sess: TensorFlow session.
      x_b_np: Numpy array. Support images.
      y_b_np: Numpy array. Support labels.
      x_b_v_np: Numpy array. Query images.
      y_b_v_np: Numpy array. Query labels.
    """
    tconfig = self.config.transfer_config
    steps = tconfig.ft_optimizer_config.max_train_steps
    batch_size = tconfig.ft_optimizer_config.batch_size
    rnd = np.random.RandomState(0)
    # Re-initialize the fast weights.
    self.reset_b(sess)
    if fdict is None:
      fdict = {}
    if batch_size == -1:
      fdict[self.inputs_b] = x_b_np
      fdict[self.labels_b] = y_b_np
    fdict[self.inputs_b_v] = x_b_v_np
    fdict[self.labels_b_v] = y_b_v_np

    cost_b_list = np.zeros([steps])
    acc_b_list = np.zeros([steps])
    acc_b_v_list = np.zeros([steps])

    # Run 1st order.
    if tconfig.ft_optimizer_config.optimizer in ['adam', 'sgd', 'mom']:
      it = six.moves.xrange(steps)
      it = tqdm(it, ncols=0, desc='solve b')
      cost_b = 0.0
      for num in it:
        if batch_size == -1:
          # Use full batch size.
          x_, y_ = x_b_np, y_b_np
        else:
          # Use mini-batch.
          assert False
          x_, y_ = self.minibatch(x_b_np, y_b_np, batch_size, rnd=rnd)
          fdict[self.inputs_b] = x_
          fdict[self.labels_b] = y_
        cost_b, acc_b_tr, acc_b_v, _ = sess.run(
            [self.cost_b, self.acc_b_tr, self.acc_b_v, self._train_op_ft],
            feed_dict=fdict)
        cost_b_list[num] = cost_b
        acc_b_list[num] = acc_b_tr
        acc_b_v_list[num] = acc_b_v
        it.set_postfix(
            cost_b='{:.3e}'.format(cost_b),
            acc_b_tr='{:.3f}'.format(acc_b_tr * 100.0),
            acc_b_v='{:.3f}'.format(acc_b_v * 100.0))
    # Run 2nd order after initial burn in.
    elif tconfig.ft_optimizer_config.optimizer in ['lbfgs']:
      # Let's use first order optimizers for now.
      assert False, 'Not supported.'
    return cost_b_list, acc_b_list, acc_b_v_list

  def solve_b(self, sess, x_b_np, y_b_np, fdict=None):
    """Solve the few-shot classifier.

    Args:
      sess: TensorFlow session.
      x_b_np: Numpy array. Support images.
      y_b_np: Numpy array. Support labels.
      fdict: Feed dict used for forward pass.
    """
    tconfig = self.config.transfer_config
    steps = tconfig.ft_optimizer_config.max_train_steps
    batch_size = tconfig.ft_optimizer_config.batch_size
    rnd = np.random.RandomState(0)
    # Re-initialize the fast weights.
    self.reset_b(sess)
    if fdict is None:
      fdict = {}
    if batch_size == -1:
      fdict[self.inputs_b] = x_b_np
      fdict[self.labels_b] = y_b_np

    # Run 1st order.
    if tconfig.ft_optimizer_config.optimizer in ['adam', 'sgd', 'mom']:
      it = six.moves.xrange(steps)
      it = tqdm(it, ncols=0, desc='solve b')
      cost_b = 0.0
      for num in it:
        if batch_size == -1:
          # Use full batch size.
          x_, y_ = x_b_np, y_b_np
        else:
          # Use mini-batch.
          assert False
          x_, y_ = self.minibatch(x_b_np, y_b_np, batch_size, rnd=rnd)
          fdict[self.inputs_b] = x_
          fdict[self.labels_b] = y_
        cost_b, acc_b_tr, _ = sess.run(
            [self.cost_b, self.acc_b_tr, self._train_op_ft], feed_dict=fdict)
        it.set_postfix(
            cost_b='{:.3e}'.format(cost_b),
            acc_b_tr='{:.3f}'.format(acc_b_tr * 100.0))

    # Run 2nd order after initial burn in.
    elif tconfig.ft_optimizer_config.optimizer in ['lbfgs']:
      step_callback = None
      loss_callback = None
      fetches = None

      scipy_interface = tconfig.scipy_interface
      if scipy_interface == 'built-in':
        self._ft_opt_scipy.minimize(
            sess,
            feed_dict=fdict,
            step_callback=step_callback,
            fetches=fetches,
            loss_callback=loss_callback)
      elif scipy_interface == 'custom':
        self._ft_opt_scipy.minimize(
            sess,
            feed_dict=fdict,
            x_key=self.inputs_b,
            x_value=x_b_np,
            y_key=self.labels_b,
            y_value=y_b_np,
            batch_size=batch_size,
            step_callback=step_callback,
            fetches=fetches,
            loss_callback=loss_callback)
      else:
        assert False, 'Unknown scipy interface'
      cost_b = sess.run(self.cost_b, feed_dict=fdict)

    return cost_b

  def rbp_b(self, sess, x_b_np, y_b_np, x_b_v_np, y_b_v_np, fdict):
    """Calculates gradients from the finetuning validation loss to the w_0.
    Will automatically restore the weights before returning.

    Args:
      sess: TensorFlow session object.
      x_b_v_np: Validation inputs of task B in numpy format.
      y_b_v_np: Validation targets of task B in numpy format.
      fdict: Feed dict used for forward pass.
    """
    fdict[self.inputs_b] = x_b_np
    fdict[self.labels_b] = y_b_np
    if self.inputs_b not in fdict:
      fdict[self.inputs_b] = x_b_np
      fdict[self.labels_b] = y_b_np
    fdict[self.inputs_b_v] = x_b_v_np
    fdict[self.labels_b_v] = y_b_v_np
    sess.run(self._update_grads_b, feed_dict=fdict)

  def reset_b(self, sess):
    """Restores the weights to its initial state."""
    sess.run(self._init_ops)

  def eval_step_b_custom_fetch(self, sess, fetches, task_b_data):
    """Evaluate one step on task B, with custom fetch."""
    fdict = self._prerun(sess, None, task_b_data)
    _ = self.solve_b(
        sess, task_b_data.x_train, task_b_data.y_train, fdict=fdict)
    return sess.run(fetches, feed_dict=fdict)

  def eval_step_b(self, sess, task_b_data):
    """Evaluate one step on task B."""
    fdict = self._prerun(sess, None, task_b_data)
    _ = self.solve_b(
        sess, task_b_data.x_train, task_b_data.y_train, fdict=fdict)
    prediction_b, y_b = sess.run([self.prediction_b_all, self.labels_b_v_all],
                                 feed_dict=fdict)
    return prediction_b, y_b

  def eval_curve_b(self, sess, task_b_data):
    """Evaluate one episode with curves."""
    fdict = self._prerun(sess, None, task_b_data)
    cost_b, acc_b, acc_b_v = self.monitor_b(
        sess,
        task_b_data.x_train,
        task_b_data.y_train,
        task_b_data.x_test,
        task_b_data.y_test,
        fdict=fdict)
    return cost_b, acc_b, acc_b_v

  def _prerun(self, sess, task_a_data, task_b_data):
    """Some steps before running."""
    fdict = self.get_fdict(task_a_data=task_a_data, task_b_data=task_b_data)
    if self.save_hidden_b:
      h_b = sess.run(
          self.hidden_b, feed_dict={self.inputs_b: task_b_data.x_train})
      fdict[self.hidden_b_plh] = h_b
    else:
      fdict[self.inputs_b] = task_b_data.x_train
    return fdict

  def train_step(self, sess, task_a_data, task_b_data):
    """Train a single step."""
    fdict = self._prerun(sess, None, task_b_data)
    cost_b = self.solve_b(
        sess, task_b_data.x_train, task_b_data.y_train, fdict=fdict)
    self.rbp_b(sess, task_b_data.x_train, task_b_data.y_train,
               task_b_data.x_test, task_b_data.y_test, fdict)
    train_op = self.train_op_b
    fdict[self.inputs] = task_a_data[0]
    fdict[self.labels] = task_a_data[1]
    cost_a, cost_b_v, _ = sess.run([self.cost_a, self.cost_b_v, train_op],
                                   feed_dict=fdict)
    return cost_a, cost_b, cost_b_v

  @property
  def backbone(self):
    """Backbone."""
    return self._backbone

  @property
  def hidden_b(self):
    """Hidden B."""
    return self._hidden_b

  @property
  def hidden_b_plh(self):
    """Hidden B placeholder."""
    return self._hidden_b_plh

  @property
  def save_hidden_b(self):
    """Whether to save hidden state in task B to avoid recomputing."""
    tconfig = self.config.transfer_config
    return (tconfig.finetune_layers == 'none' and
            tconfig.ft_optimizer_config.batch_size == -1)

  @property
  def prediction_b(self):
    """Prediction on task B."""
    return self._prediction_b

  @property
  def prediction_b_all(self):
    """All prediction on task B."""
    return self._prediction_b_all

  @property
  def acc_b_tr(self):
    """Accuracy on task B support."""
    return self._acc_b_tr

  @property
  def acc_b_v(self):
    """Accuracy on task B query."""
    return self._acc_b_v

  @property
  def train_op(self):
    """Overall training op."""
    return self._train_op

  @property
  def train_op_ft(self):
    """Training op for learning task B support set."""
    return self._train_op_ft

  @property
  def train_op_a(self):
    """Training op on task A."""
    return self._train_op_a

  @property
  def train_op_b(self):
    """Training op on task B."""
    return self._train_op_b

  @property
  def config(self):
    """Model config."""
    return self._config

  @property
  def learn_rate(self):
    """Learning rate."""
    return self._learn_rate

  @property
  def num_classes_a(self):
    """Number of classes on task A."""
    return self._num_classes_a

  @property
  def num_classes_b(self):
    """Number of classes on task B."""
    return self._num_classes_b
