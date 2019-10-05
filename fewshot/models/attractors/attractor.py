"""A common abstract class for attractor based regularization functions."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

ATTRACTOR_REGISTRY = {}


def RegisterAttractor(attractor_name):
  """Registers an attractor class."""

  def decorator(f):
    ATTRACTOR_REGISTRY[attractor_name] = f
    return f

  return decorator


def get_attractor(attractor_name, config):
  """Initialize an attractor class."""
  if attractor_name in ATTRACTOR_REGISTRY:
    return ATTRACTOR_REGISTRY[attractor_name](config)
  else:
    return None


class Attractor(object):
  """Attractor base class."""

  def __init__(self, config=None):
    """Initialize the attractor with a configuration object."""
    self._config = config

  def __call__(self, fast_weights, **kwargs):
    raise NotImplementedError()

  def combine_wb(self, w, b):
    """Combine weights and biases."""
    return tf.concat([w, tf.expand_dims(b, 0)], axis=0)

  @property
  def config(self):
    return self._config
