"""Feature extractor backbone interface.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.resnet_base import ResnetBase
from fewshot.models.model_factory import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("resnet_backbone")
class ResnetBackbone(ResnetBase):
  """Defines the interface of a backbone network,"""

  def __call__(self,
               x,
               is_training=True,
               ext_wts=None,
               reuse=None,
               slow_bn=False,
               exit='top'):
    """See Backbone class for documentation."""
    config = self.config
    assert config.version in ['v1', 'v2', 'snailv1',
                              'snailv2'], 'Unknown version'
    dtype = self.dtype
    self._slow_bn = slow_bn
    self._ext_wts = ext_wts
    # assert ext_wts is None  # Enable this in the future.
    self._bn_update_ops = []
    self._exit = exit
    with tf.variable_scope("phi", reuse=reuse):
      if self.config.data_format == 'NCHW':
        x = tf.transpose(x, [0, 3, 1, 2])
      h = self.build_inference_network(x, is_training=is_training)

    # Store the weights in a dictionary.
    if self._weights is None:
      all_vars = tf.global_variables()
      all_vars = list(filter(lambda x: 'phi' in x.name, all_vars))
      self._weights = dict(
          zip(map(lambda x: x.name.split(':')[0], all_vars), all_vars))
      log.info('Created variable dictionary')
      log.info(self._weights)
    return h

  def build_inference_network(self, x, is_training):
    config = self.config
    num_stages = len(self.config.num_residual_units)
    strides = config.strides
    filters = [ff for ff in config.num_filters]  # Copy filter config.
    h = self._init_conv(x, filters[0], is_training=is_training)
    if config.use_bottleneck:
      res_func = self._bottleneck_residual
      # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]
      for ii in range(1, len(filters)):
        filters[ii] *= 4
    elif config.version == 'snailv1':
      res_func = self._residual2
    elif config.version == 'snailv2':
      res_func = self._residual3
    else:
      res_func = self._residual

    # New version, single for-loop. Easier for checkpoint.
    nlayers = sum(config.num_residual_units)
    ss = 0
    ii = 0
    for ll in range(nlayers):
      # Residual unit configuration.
      if ss == 0 and ii == 0 and self.config.version == "v2":
        no_activation = True
      else:
        no_activation = False
      if ii == 0:
        in_filter = filters[ss]
        stride = self._stride_arr(strides[ss])
      else:
        in_filter = filters[ss + 1]
        stride = self._stride_arr(1)
      out_filter = filters[ss + 1]

      # Build residual unit.
      with tf.variable_scope("unit_{}_{}".format(ss + 1, ii)):
        add_relu = True
        if not config.add_last_relu:
          if ll == nlayers - 1:
            add_relu = False
        # assert config.add_last_relu
        # assert add_relu
        h = res_func(
            h,
            in_filter,
            out_filter,
            stride,
            no_activation=no_activation,
            is_training=is_training,
            add_relu=add_relu)

      if (ii + 1) % config.num_residual_units[ss] == 0:
        ss += 1
        ii = 0
      else:
        ii += 1

      if self._exit == 'bottom':
        assert num_stages > 3  # Only ImageNet model can exit at bottom.

      if self._exit == 'middle':
        if ss == num_stages - 1:
          log.error(h.name)
          log.error('Early exit middle')
          break
      elif self._exit == 'midbottom':
        if ss == num_stages - 2:
          log.error(h.name)
          log.error('Early exit midbottom')
          break
      elif self._exit == 'bottom':
        if ss == num_stages - 3:
          log.error(h.name)
          log.error('Early exit bottom')
          break
      elif self._exit == 'top':
        pass
      else:
        raise ValueError('Unknown exit {}'.format(self._exit))

    if config.version == 'snailv1':
      with tf.variable_scope("unit_last"):
        h = self._conv("conv1", h, 1, self.config.num_filters[-1], 2048,
                       self._stride_arr(1))
        assert h.shape[2] == 6, str(h.shape)
        h = self._global_avg_pool(h)
        if config.add_last_relu:
          assert False
          h = self._relu("final_relu", h)
        if is_training:
          h = tf.nn.dropout(h, keep_prob=0.9)
        h = self._fully_connected(h, 384)

    elif config.version == 'snailv2':
      with tf.variable_scope("unit_last"):
        assert h.shape[2] == 6, str(h.shape)
        h = self._global_avg_pool(h, keep_dims=True)
        h = self._normalize('bn1', h, is_training=is_training)
        h = tf.nn.relu(h)
        h = self._conv("conv1", h, 1, self.config.num_filters[-1], 384,
                       self._stride_arr(1))
        if is_training:
          h = tf.nn.dropout(h, keep_prob=0.9)
        h = self._normalize('bn2', h, is_training=is_training)
        h = tf.nn.relu(h)
        if self.config.data_format == 'NCHW':
          h = tf.squeeze(h, [2, 3])
        else:
          h = tf.squeeze(h, [1, 2])
        h = self._fully_connected(h, 512)
        if is_training:
          h = tf.nn.dropout(h, keep_prob=0.9)
    else:
      # Only top layer needs the final BN and ReLU.
      if self._exit == 'top' and self.config.version == "v2":
        with tf.variable_scope("unit_last"):
          h = self._normalize("final_bn", h, is_training=is_training)
          h = self._relu("final_relu", h)

      if config.global_avg_pool:
        h = self._global_avg_pool(h)

    return h
