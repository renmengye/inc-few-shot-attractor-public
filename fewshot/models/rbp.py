"""Recurrent backprop.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import tensorflow as tf

from fewshot.utils.logger import get as get_logger

log = get_logger()


def rbp(x, h1, h2, f, nstep, lbd=0.9, debug=False):
  """Recurrent backprop.

  Args:
    x: Inputs to the dynamical process.
    h1: List of final hidden state.
    h2: List of second last hidden state.
    f: Final cost, function to optimize.
    nstep: Number of recurrent backprop steps.
    lbd: Damping constant, default 0.9.
    debug: Whether to print out intermediate values.

  Returns:
    grad_x: Gradients of f wrt. x.

  Note: You should only unroll the graph once."""
  if type(h1) != list:
    h1 = [h1]
  if type(h2) != list:
    h2 = [h2]
  if type(x) != list:
    x = [x]
  assert lbd >= 0.0

  grad_h = tf.gradients(f, h1, gate_gradients=1)
  nv = [tf.stop_gradient(_) for _ in grad_h]
  ng = [tf.stop_gradient(_) for _ in grad_h]

  for step in six.moves.xrange(nstep):
    j_nv = tf.gradients(h1, h2, grad_ys=nv, gate_gradients=1)
    if lbd > 0.0:
      nv = [j_nv_ - lbd * nv_ for j_nv_, nv_ in zip(j_nv, nv)]
    else:
      nv = j_nv
    if debug:
      # Debug mode, print ng values.
      ng_norm = tf.add_n([tf.sqrt(tf.reduce_sum(tf.square(_))) for _ in ng])
      nv_norm = tf.add_n([tf.sqrt(tf.reduce_sum(tf.square(_))) for _ in nv])
      print_ng = tf.Print(tf.constant(0.0), ['ng norm', step, ng_norm])
      print_nv = tf.Print(tf.constant(0.0), ['nv norm', step, nv_norm])
      with tf.control_dependencies([print_ng, print_nv]):
        nv = [tf.identity(_) for _ in nv]
        ng = [tf.identity(_) for _ in ng]
    ng = [ng_ + nv_ for ng_, nv_ in zip(ng, nv)]
  grad = tf.gradients(h1, x, grad_ys=ng, gate_gradients=1)
  return grad
