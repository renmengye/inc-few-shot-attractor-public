"""A basic feature extractor.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.backbone import Backbone
from fewshot.models.model_factory import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel('fc_backbone')
class FCBackbone(Backbone):
  """A fully connected backbone for testing."""

  def __call__(self, x, is_training=True, ext_wts=None, reuse=None, **kwargs):
    """See Backbone class for documentation."""
    x_ = tf.reshape(
        x,
        [-1, self.config.height * self.config.width * self.config.num_channel])
    return x_
