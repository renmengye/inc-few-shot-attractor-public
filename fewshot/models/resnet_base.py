from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import weight_variable
from fewshot.models.backbone import Backbone
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ResnetBase(Backbone):

  def _batch_norm_slow(self, name, x, is_training=True):
    """A slow version of batch normalization, allows self-defined variables."""
    if self.config.data_format == 'NCHW':
      axis = 1
      axes = [0, 2, 3]
    else:
      axis = -1
      axes = [0, 1, 2]

    shape = [int(x.get_shape()[axis])]
    vs = 'BatchNorm'

    decay = 0.999
    eps = 0.001

    def get_vars(vs):
      with tf.variable_scope(vs):
        beta = self._weight_variable(
            shape,
            name="beta",
            init_method='constant',
            init_param={'val': 0.0})
        gamma = self._weight_variable(
            shape,
            name="gamma",
            init_method='constant',
            init_param={'val': 1.0})
        emean = self._weight_variable(
            shape,
            name='moving_mean',
            trainable=False,
            dtype=x.dtype,
            init_method='constant',
            init_param={'val': 0.0})
        evar = self._weight_variable(
            shape,
            name='moving_variance',
            trainable=False,
            dtype=x.dtype,
            init_method='constant',
            init_param={'val': 0.0})
      mean, var = tf.nn.moments(x, axes=axes)
      if self.config.data_format == 'NCHW':
        gamma_ = tf.reshape(gamma, [1, -1, 1, 1])
        beta_ = tf.reshape(beta, [1, -1, 1, 1])
        mean_ = tf.reshape(mean, [1, -1, 1, 1])
        var_ = tf.reshape(var, [1, -1, 1, 1])
        emean_ = tf.reshape(emean, [1, -1, 1, 1])
        evar_ = tf.reshape(evar, [1, -1, 1, 1])
      else:
        gamma_ = gamma
        beta_ = beta
        mean_ = mean
        var_ = var
        emean_ = emean
        evar_ = evar
      ema_mean_op = tf.assign_sub(emean, (emean - mean) * (1 - decay))
      ema_var_op = tf.assign_sub(evar, (evar - var) * (1 - decay))
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_mean_op)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_var_op)
      return mean_, var_, emean_, evar_, beta_, gamma_

    try:
      mean_, var_, emean_, evar_, beta_, gamma_ = get_vars('BatchNorm')
    except Exception as e:
      log.error(e)
      log.error('Try another batch norm')
      mean_, var_, emean_, evar_, beta_, gamma_ = get_vars('BatchNorm_1')
    if is_training:
      return tf.nn.batch_normalization(x, mean_, var_, beta_, gamma_, eps)
    else:
      return tf.nn.batch_normalization(x, emean_, evar_, beta_, gamma_, eps)

  def _batch_norm(self, name, x, is_training=True):
    """Batch normalization."""
    if self._slow_bn:
      return self._batch_norm_slow(name, x, is_training=is_training)
    else:
      return tf.contrib.layers.batch_norm(
          x,
          fused=True,
          data_format=self.config.data_format,
          is_training=is_training,
          scale=True)

  def _group_norm_slow(self, name, x, is_training=True):
    """Slow version of group norm, allow self defined variables."""
    with tf.variable_scope(name):
      if self.config.data_format == 'NCHW':
        channels_axis = 1
        reduction_axes = (2, 3)
      elif self.config.data_format == 'NHWC':
        channels_axis = 3
        reduction_axes = (1, 2)
      x_shape = tf.shape(x)
      x_shape_list = [-1, x_shape[1], x_shape[2], x_shape[3]]
      axes_before_channel = x_shape_list[:channels_axis]
      axes_after_channel = x_shape_list[channels_axis + 1:]
      G = self.config.num_norm_groups
      C = int(x.shape[channels_axis])
      shape_after = axes_before_channel + [G, C // G] + axes_after_channel
      x_reshape = tf.reshape(x, shape_after)
      moment_axes = [channels_axis + 1]
      for a in reduction_axes:
        if a > channels_axis:
          moment_axes.append(a + 1)
        else:
          moment_axes.append(a)

      beta = self._weight_variable([C],
                                   name="beta",
                                   init_method='constant',
                                   init_param={'val': 0.0})
      gamma = self._weight_variable([C],
                                    name="gamma",
                                    init_method='constant',
                                    init_param={'val': 1.0})
      beta_shape = [1, 1, 1, 1, 1]
      beta_shape[channels_axis] = G
      beta_shape[channels_axis + 1] = C // G
      beta_reshape = tf.reshape(beta, beta_shape)
      gamma_reshape = tf.reshape(gamma, beta_shape)
      mean, variance = tf.nn.moments(x_reshape, moment_axes, keep_dims=True)
      log.info('Moment axes {}'.format(moment_axes))
      log.info(variance.shape)
      log.info(x_reshape.shape)
      log.info(mean.shape)
      epsilon = 1e-6
      gain = tf.rsqrt(variance + epsilon)
      offset = -mean * gain
      gain *= gamma_reshape
      offset *= gamma_reshape
      offset += beta_reshape
      normed = x_reshape * gain + offset
      return tf.reshape(normed, x_shape)

  def _group_norm(self, name, x, is_training=True):
    # if self._slow_bn:
    return self._group_norm_slow(name, x, is_training=is_training)
    if self.config.data_format == 'NCHW':
      channels_axis = -3
      reduction_axes = (-2, -1)
    elif self.config.data_format == 'NHWC':
      channels_axis = -1
      reduction_axes = (-3, -2)
    # print(x, x.name)
    normed = tf.contrib.layers.group_norm(
        x,
        groups=self.config.num_norm_groups,
        channels_axis=channels_axis,
        reduction_axes=reduction_axes,
        trainable=is_training,
        scope=name)
    return normed

  def _normalize(self, name, x, is_training=True):
    """Normalize the activations"""
    if self.config.normalization == "batch_norm":
      return self._batch_norm(name, x, is_training=is_training)
    elif self.config.normalization == "group_norm":
      return self._group_norm(name, x, is_training=is_training)

  def _possible_downsample(self, x, in_filter, out_filter, stride):
    """Downsample the feature map using average pooling, if the filter size
    does not match."""
    if stride[2] > 1:
      with tf.variable_scope("downsample"):
        x = tf.nn.avg_pool(
            x,
            stride,
            stride,
            padding="SAME",
            data_format=self.config.data_format)

    if in_filter < out_filter:
      pad_ = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
      with tf.variable_scope("pad"):
        if self.config.data_format == 'NHWC':
          x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_])
        else:
          x = tf.pad(x, [[0, 0], pad_, [0, 0], [0, 0]])
    return x

  def _possible_bottleneck_downsample(self,
                                      x,
                                      in_filter,
                                      out_filter,
                                      stride,
                                      is_training=True):
    """Downsample projection layer, if the filter size does not match."""
    if stride[2] > 1 or in_filter != out_filter:
      x = self._conv("project", x, 1, in_filter, out_filter, stride)
      if self.config.version == "v1":
        x = self._normalize("project_bn", x, is_training=is_training)
    return x

  def _residual_inner(self,
                      x,
                      in_filter,
                      out_filter,
                      stride,
                      no_activation=False,
                      is_training=True):
    """Transformation applied on residual units."""
    if self.config.version == "v2":
      with tf.variable_scope("sub1"):
        if not no_activation:
          x = self._normalize("bn1", x, is_training=is_training)
          x = self._relu("relu1", x)
        x = self._conv("conv1", x, 3, in_filter, out_filter, stride)
      with tf.variable_scope("sub2"):
        x = self._normalize("bn2", x, is_training=is_training)
        x = self._relu("relu2", x)
        x = self._conv("conv2", x, 3, out_filter, out_filter,
                       self._stride_arr(1))
    else:
      with tf.variable_scope("sub1"):
        x = self._conv("conv2", x, 3, in_filter, out_filter, stride)
        x = self._normalize("bn1", x, is_training=is_training)
        x = self._relu("relu1", x)
      with tf.variable_scope("sub2"):
        x = self._conv("conv2", x, 3, out_filter, out_filter,
                       self._stride_arr(1))
        x = self._normalize("bn2", x, is_training=is_training)
    return x

  def _bottleneck_residual_inner(self,
                                 x,
                                 in_filter,
                                 out_filter,
                                 stride,
                                 no_activation=False,
                                 is_training=True):
    """Transformation applied on bottleneck residual units."""
    if self.config.version == "v2":
      with tf.variable_scope("sub1"):
        if not no_activation:
          x = self._normalize("bn1", x, is_training=is_training)
          x = self._relu("relu1", x)
        x = self._conv("conv1", x, 1, in_filter, out_filter // 4, stride)
      with tf.variable_scope("sub2"):
        x = self._normalize("bn2", x, is_training)
        x = self._relu("relu2", x)
        x = self._conv("conv2", x, 3, out_filter // 4, out_filter // 4,
                       self._stride_arr(1))
      with tf.variable_scope("sub3"):
        x = self._normalize("bn3", x, is_training=is_training)
        x = self._relu("relu3", x)
        x = self._conv("conv3", x, 1, out_filter // 4, out_filter,
                       self._stride_arr(1))
    elif self.config.version == "v1":
      with tf.variable_scope("sub1"):
        x = self._conv("conv1", x, 1, in_filter, out_filter // 4, stride)
        x = self._normalize("bn1", x, is_training=is_training)
        x = self._relu("relu1", x)
      with tf.variable_scope("sub2"):
        x = self._conv("conv2", x, 3, out_filter // 4, out_filter // 4,
                       self._stride_arr(1))
        x = self._normalize("bn1", x, is_training=is_training)
        x = self._relu("relu1", x)
      with tf.variable_scope("sub3"):
        x = self._conv("conv3", x, 1, out_filter // 4, out_filter,
                       self._stride_arr(1))
        x = self._normalize("bn3", x, is_training=is_training)
    else:
      raise ValueError("Unkonwn version")
    return x

  def _residual_inner2(self,
                       x,
                       in_filter,
                       out_filter,
                       stride,
                       no_activation=False,
                       is_training=True):
    """Transformation applied on residual units."""
    # This is SNAIL Resnet
    with tf.variable_scope("sub1"):
      x = self._conv("conv1", x, 3, in_filter, out_filter, self._stride_arr(1))
      x = self._normalize("bn1", x, is_training=is_training)
      x = self._relu("relu1", x)
    with tf.variable_scope("sub2"):
      x = self._conv("conv2", x, 3, out_filter, out_filter,
                     self._stride_arr(1))
      x = self._normalize("bn2", x, is_training=is_training)
      x = self._relu("relu2", x)
    with tf.variable_scope("sub3"):
      x = self._conv("conv3", x, 3, out_filter, out_filter,
                     self._stride_arr(1))
      x = self._normalize("bn3", x, is_training=is_training)
    return x

  def _residual(self,
                x,
                in_filter,
                out_filter,
                stride,
                no_activation=False,
                is_training=True,
                add_relu=True):
    """Residual unit with 2 sub layers."""
    orig_x = x
    x = self._residual_inner(
        x,
        in_filter,
        out_filter,
        stride,
        no_activation=no_activation,
        is_training=is_training)
    x += self._possible_downsample(orig_x, in_filter, out_filter, stride)
    if self.config.version == "v1" and add_relu:
      x = self._relu("relu3", x)
    # x = debug_identity(x)
    return x

  def _residual2(self,
                 x,
                 in_filter,
                 out_filter,
                 stride,
                 no_activation=False,
                 is_training=True,
                 add_relu=True):
    """Residual unit with 2 sub layers."""
    orig_x = self._conv("proj", x, 1, in_filter, out_filter,
                        self._stride_arr(1))
    x = self._residual_inner2(
        x,
        in_filter,
        out_filter,
        self._stride_arr(1),
        no_activation=no_activation,
        is_training=is_training)
    x = tf.nn.max_pool(
        x + orig_x,
        self._stride_arr(2),
        stride,
        padding='SAME',
        data_format=self.config.data_format)
    if is_training:
      x = tf.nn.dropout(x, keep_prob=0.9)
    return x

  def _residual3(self,
                 x,
                 in_filter,
                 out_filter,
                 stride,
                 no_activation=False,
                 is_training=True,
                 add_relu=True):
    """Residual unit with 2 sub layers."""
    orig_x = self._conv("proj", x, 1, in_filter, out_filter,
                        self._stride_arr(1))
    x = self._residual_inner2(
        x,
        in_filter,
        out_filter,
        self._stride_arr(1),
        no_activation=no_activation,
        is_training=is_training)
    x = tf.nn.max_pool(
        x + orig_x,
        self._stride_arr(2),
        stride,
        padding='SAME',
        data_format=self.config.data_format)
    return x

  def _bottleneck_residual(self,
                           x,
                           in_filter,
                           out_filter,
                           stride,
                           no_activation=False,
                           is_training=True,
                           add_relu=True):
    """Bottleneck resisual unit with 3 sub layers."""
    orig_x = x
    x = self._bottleneck_residual_inner(
        x,
        in_filter,
        out_filter,
        stride,
        no_activation=no_activation,
        is_training=is_training)
    x += self._possible_bottleneck_downsample(
        orig_x, in_filter, out_filter, stride, is_training=is_training)
    if self.config.version == "v1" and add_relu:
      x = self._relu("relu3", x)
    return x

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      if self.config.filter_initialization == "normal":
        n = filter_size * filter_size * out_filters
        init_method = "truncated_normal"
        init_param = {"mean": 0, "stddev": np.sqrt(2.0 / n)}
      elif self.config.filter_initialization == "uniform":
        init_method = "uniform_scaling"
        init_param = {"factor": 1.0}
      kernel = self._weight_variable(
          [filter_size, filter_size, in_filters, out_filters],
          init_method=init_method,
          init_param=init_param,
          wd=self.config.wd,
          dtype=self.dtype,
          name="w")
      return tf.nn.conv2d(
          x,
          kernel,
          strides,
          padding="SAME",
          data_format=self.config.data_format)

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x_shape = x.get_shape()
    d = x_shape[1]
    w = self._weight_variable(
        [d, out_dim],
        init_method="uniform_scaling",
        # init_param={"factor": 1.0},
        init_param={"factor": 1 / np.sqrt(float(out_dim))},
        wd=self.config.wd,
        dtype=self.dtype,
        name="w")
    b = self._weight_variable([out_dim],
                              init_method="constant",
                              init_param={"val": 0.0},
                              name="b",
                              dtype=self.dtype)
    return tf.nn.xw_plus_b(x, w, b)

  def _weight_variable(self,
                       shape,
                       init_method=None,
                       dtype=tf.float32,
                       init_param=None,
                       wd=None,
                       name=None,
                       trainable=True,
                       seed=0):
    """Wrapper to declare variables. Default on CPU."""
    if self._ext_wts is None:
      return weight_variable(
          shape,
          init_method=init_method,
          dtype=dtype,
          init_param=init_param,
          wd=wd,
          name=name,
          trainable=trainable,
          seed=seed)
    else:
      assert self._slow_bn, "Must enable slow BN"
      assert name is not None  # Use name to retrieve the variable name
      vs = tf.get_variable_scope()
      var_name = vs.name + '/' + name
      if var_name in self._ext_wts:
        log.info('Found variable {} in external weights'.format(var_name))
        return self._ext_wts[var_name]
      else:
        log.error('Not found variable {} in external weights'.format(var_name))
        raise ValueError('Variable not found')

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self.config.data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _relu(self, name, x):
    if self.config.leaky_relu > 0.0:
      return tf.nn.leaky_relu(x, alpha=self.config.leaky_relu, name=name)
    else:
      return tf.nn.relu(x, name=name)

  def _init_conv(self, x, n_filters, is_training=True):
    """Build initial conv layers."""
    config = self.config
    init_filter = config.init_filter
    with tf.variable_scope("init"):
      h = self._conv("init_conv", x, init_filter, self.config.num_channel,
                     n_filters, self._stride_arr(config.init_stride))
      h = self._normalize("init_bn", h, is_training=is_training)
      h = self._relu("init_relu", h)
      # Max-pooling is used in ImageNet experiments to further reduce
      # dimensionality.
      if config.init_max_pool:
        h = tf.nn.max_pool(
            h,
            self._stride_arr(3),
            self._stride_arr(2),
            padding="SAME",
            data_format=self.config.data_format)
    return h

  def _global_avg_pool(self, x, keep_dims=False):
    if self.config.data_format == 'NCHW':
      return tf.reduce_mean(x, [2, 3], keep_dims=keep_dims)
    else:
      return tf.reduce_mean(x, [1, 2], keep_dims=keep_dims)
