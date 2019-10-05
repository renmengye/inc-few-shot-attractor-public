"""
Checkpointing utilities.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf


def build_checkpoint_with_initializer(var_list, scope="checkpoint"):
  """Use placeholder as checkpoint."""
  with tf.variable_scope(scope):

    def get_var(x):
      x_name = x.name.split(":")[0]
      plh = tf.placeholder(x.dtype, x.get_shape(), name=x_name + '_init')
      return tf.get_variable(x_name, None, x.dtype, plh, trainable=False), plh

    ckpt_with_initializer = list(map(get_var, var_list))
  return ckpt_with_initializer


def build_checkpoint(var_list, scope="checkpoint"):
  """Build checkpoint variables."""
  with tf.variable_scope(scope):

    def get_var(x):
      return tf.get_variable(
          x.name.split(":")[0],
          x.get_shape(),
          x.dtype,
          tf.constant_initializer(0, dtype=x.dtype),
          trainable=False)

    ckpt = list(map(get_var, var_list))
  return ckpt


def read_checkpoint(ckpt, var_list, use_locking=False):
  """Read from the checkpoint."""
  return tf.group(*[
      tf.assign(ww, ck, use_locking=use_locking)
      for ck, ww in zip(ckpt, var_list)
  ])


def write_checkpoint(ckpt, var_list, use_locking=False):
  """Write to the checkpoint."""
  assert len(ckpt) == len(var_list)
  return tf.group(*[
      tf.assign(ck, ww, use_locking=use_locking)
      for ck, ww in zip(ckpt, var_list)
  ])


def update_add_checkpoint(ckpt, var_list, use_locking=False):
  """Add to the checkpoint."""
  assert len(ckpt) == len(var_list)
  return tf.group(*[
      tf.assign_add(ck, ww, use_locking=use_locking)
      for ck, ww in zip(ckpt, var_list)
  ])


def clear_checkpoint(ckpt):
  return tf.variables_initializer(ckpt)
