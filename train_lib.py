"""Training utilities."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fewshot.data.batch_iter import BatchIterator
from fewshot.data.data_factory import get_concurrent_iterator
from fewshot.utils.logger import get as get_logger

log = get_logger()


def get_metadata(dataset):
  """Gets dataset metadata.

  Args:
    dataset: String. Name of the dataset.
  """
  num_unlabel = 0
  if dataset == 'mini-imagenet':
    num_classes_a = 64
    num_classes_b = 64
    num_classes_b_val = 16
    num_classes_b_test = 20
  elif dataset == 'tiered-imagenet':
    num_classes_a = 200
    num_classes_b = 151
    num_classes_b_val = 97
    num_classes_b_test = 160
  else:
    raise Exception('')

  if dataset == 'mini-imagenet':
    trainsplit_a_train = 'train_phase_train'
    trainsplit_a_val = 'train_phase_val'
    trainsplit_a_test = 'train_phase_test'
    trainsplit_b = 'train_phase_train'
    label_ratio = 1.0
    image_split_file_a_train = None
    image_split_file_a_val = None
    image_split_file_a_test = None
    image_split_file_b = None
  elif dataset == 'tiered-imagenet':
    trainsplit_a_train = 'train_a'
    trainsplit_a_val = 'train_a'
    trainsplit_a_test = 'train_a'
    trainsplit_b = 'train_b'
    image_split_file_a_train = 'fewshot/data/tiered_imagenet_split/train_a_phase_train.csv'  # NOQA
    image_split_file_a_val = 'fewshot/data/tiered_imagenet_split/train_a_phase_val.csv'  # NOQA
    image_split_file_a_test = 'fewshot/data/tiered_imagenet_split/train_a_phase_test.csv'  # NOQA
    image_split_file_b = None
    label_ratio = 1.0
  else:
    assert False

  log.info('Using train split A train {}'.format(trainsplit_a_train))
  log.info('Using train split A val {}'.format(trainsplit_a_val))
  log.info('Using train split A test {}'.format(trainsplit_a_test))
  log.info('Using train split B {}'.format(trainsplit_b))

  return {
      'num_unlabel': num_unlabel,
      'num_classes_a': num_classes_a,
      'num_classes_b': num_classes_b,
      'num_classes_b_val': num_classes_b_val,
      'num_classes_b_test': num_classes_b_test,
      'trainsplit_a_train': trainsplit_a_train,
      'trainsplit_a_val': trainsplit_a_val,
      'trainsplit_a_test': trainsplit_a_test,
      'trainsplit_b': trainsplit_b,
      'label_ratio': label_ratio,
      'image_split_file_a_train': image_split_file_a_train,
      'image_split_file_a_val': image_split_file_a_val,
      'image_split_file_a_test': image_split_file_a_test,
      'image_split_file_b': image_split_file_b
  }


def get_iter(size,
             get_fn,
             batch_size,
             cycle=True,
             shuffle=True,
             max_queue_size=50,
             num_threads=5,
             seed=0):
  b = BatchIterator(
      size,
      batch_size=batch_size,
      cycle=True,
      shuffle=True,
      get_fn=get_fn,
      log_epoch=-1,
      seed=seed)
  if num_threads > 1:
    return get_concurrent_iterator(
        b, max_queue_size=max_queue_size, num_threads=num_threads)
  else:
    return b
