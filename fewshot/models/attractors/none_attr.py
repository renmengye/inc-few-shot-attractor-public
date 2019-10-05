"""None attractor."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.attractors.attractor import Attractor
from fewshot.models.attractors.attractor import RegisterAttractor


@RegisterAttractor('none')
class NoneAttractor(Attractor):

  def __call__(self, fast_weights, reuse=None, **kwargs):
    return tf.constant(0.0)
