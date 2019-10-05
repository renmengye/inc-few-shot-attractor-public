import tensorflow as tf
import os


def debug_identity(x, name=None):
  if int(os.environ.get('TF_DEBUG', 0)) == 1:
    return tf.Print(
        x, [
            x.name if name is None else name,
            tf.reduce_mean(x),
            tf.reduce_max(x),
            tf.reduce_min(x),
            tf.shape(x)
        ],
        summarize=25)
  else:
    return x
