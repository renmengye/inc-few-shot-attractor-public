from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import tensorflow as tf

from fewshot.data.concurrent_batch_iter import ConcurrentBatchIterator

flags = tf.flags
flags.DEFINE_string("data_root", "data", "Data root")
flags.DEFINE_string("data_folder", None, "Data folder")
FLAGS = tf.flags.FLAGS

DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
  """Registers a dataset class"""

  def decorator(f):
    DATASET_REGISTRY[dataset_name] = f
    return f

  return decorator


def get_data_folder(dataset_name):
  if FLAGS.data_folder is None:
    data_folder = os.path.join(FLAGS.data_root, dataset_name)
  else:
    data_folder = FLAGS.data_folder
  return data_folder


def get_dataset(dataset_name, split, *args, **kwargs):
  if dataset_name in DATASET_REGISTRY:
    return DATASET_REGISTRY[dataset_name](get_data_folder(dataset_name), split,
                                          *args, **kwargs)
  else:
    raise ValueError("Unknown dataset \"{}\"".format(dataset_name))


def get_concurrent_iterator(dataset, max_queue_size=100, num_threads=10):
  return ConcurrentBatchIterator(
      dataset,
      max_queue_size=max_queue_size,
      num_threads=num_threads,
      log_queue=-1)
