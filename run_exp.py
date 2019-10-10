"""Experiments for incremental few-shot learning.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
python run_exp.py --config {CONFIG_FILE}                     \
                  --dataset {DATASET}                        \
                  --pretrain {PRETRAIN_CKPT_FOLDER}          \
                  --nshot {NUMBER_OF_SHOTS}                  \
                  --nclasses_b {NUMBER_OF_FEWSHOT_WAYS}      \
                  --result {SAVE_FOLDER}                     \
                  --tag {EXPERIMENT_NAME}                    \
                  [--eval]                                   \
                  [--retest]
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import six
import sys
import tensorflow as tf
import time

from tqdm import tqdm
from google.protobuf.text_format import Merge, MessageToString

from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.data.batch_iter import BatchIterator
from fewshot.data.data_factory import get_concurrent_iterator
from fewshot.data.data_factory import get_dataset
from fewshot.data.mini_imagenet import MiniImageNetDataset  # NOQA
from fewshot.data.tiered_imagenet import TieredImageNetDataset  # NOQA
from fewshot.models.multi_task_model import MultiTaskModel
from fewshot.models.imprint_model import ImprintModel
from fewshot.models.attractor_model import AttractorModel
from fewshot.models.attractor_model_bptt import AttractorModelBPTT
from fewshot.utils import logger
from train_lib import get_metadata

log = logger.get()

flags = tf.flags
flags.DEFINE_bool("eval", False, "Whether run evaluation only")
flags.DEFINE_bool("retest", False, "Reload everything")
flags.DEFINE_bool("test", True, "Whether to run test")
flags.DEFINE_bool("val", True, "Whether to run val")
flags.DEFINE_integer("nclasses_a", -1, "Number of classes for pretraining")
flags.DEFINE_integer("nclasses_b", 5, "Number of classes for few-shot")
flags.DEFINE_integer("nepisode", 600, "Number of evaluation episodes")
flags.DEFINE_integer("nepisode_final", 2000, "Number of evaluation episodes")
flags.DEFINE_integer("nshot", 1, "nshot")
flags.DEFINE_integer("ntest", 5, "Number of test images per episode")
flags.DEFINE_string("config", None, "Experiment config file")
flags.DEFINE_string("dataset", "omniglot", "Dataset name")
flags.DEFINE_string("pretrain", None, "Restore checkpoint name")
flags.DEFINE_string("results", "./results", "Save folder")
flags.DEFINE_string("tag", None, "Experiment tag")
FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.ERROR)


def get_exp_logger(log_folder):
  """Gets a TensorBoard logger."""
  with tf.name_scope('Summary'):
    writer = tf.summary.FileWriter(log_folder)

  class ExperimentLogger():

    def log(self, name, niter, value):
      summary = tf.Summary()
      summary.value.add(tag=name, simple_value=float(value))
      writer.add_summary(summary, niter)

    def flush(self):
      """Flushes results to disk."""
      writer.flush()

    def close(self):
      """Closes writer."""
      writer.close()

  return ExperimentLogger()


def get_saver(log_folder):
  saver = tf.train.Saver()

  class Saver():

    def get_session(self, sess):
      session = sess
      while type(session).__name__ != 'Session':
        session = session._sess
      return session

    def save(self, sess, tag):
      saver.save(
          self.get_session(sess),
          os.path.join(log_folder, 'ckpt-{}'.format(tag)))

  return Saver()


def save_config(config, save_folder):
  """Saves configuration to a file."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "config.prototxt")
  with open(config_file, "w") as f:
    f.write(MessageToString(config))
  cmd_file = os.path.join(save_folder, "cmd-{}.txt".format(time.time()))
  if not os.path.exists(cmd_file):
    with open(cmd_file, "w") as f:
      f.write(' '.join(sys.argv))


def get_iter(size,
             get_fn,
             batch_size,
             cycle=True,
             shuffle=True,
             max_queue_size=50,
             num_threads=5,
             seed=0):
  """Gets data iterator."""
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


def top1(pred, label):
  """Calculates top 1 accuracy."""
  assert pred.shape[0] == label.shape[0], '{} != {}'.format(
      pred.shape[0], label.shape[0])
  return (np.argmax(pred, axis=1) == label).mean()


def topk(pred, label, k):
  """Calculates top 5 accuracy."""
  assert pred.shape[0] == label.shape[0], '{} != {}'.format(
      pred.shape[0], label.shape[0])
  topk_choices = np.argsort(pred, axis=1)[:, ::-1][:, :k]
  return np.sum(topk_choices == np.expand_dims(label, 1), axis=1).mean()


def stderr(array):
  """Calculates standard error."""
  return array.std() / np.sqrt(float(array.size))


def evaluate_a(sess, model, task_it, num_steps):
  """Evaluate the model on task A."""
  acc_list = np.zeros([num_steps])
  acc_top5_list = np.zeros([num_steps])
  it = tqdm(six.moves.xrange(num_steps), ncols=0)
  for tt in it:
    task_data = task_it.next()
    prediction_a, labels_a = model.eval_step_a(sess, task_data)
    acc_list[tt] = top1(prediction_a, labels_a)
    acc_top5_list[tt] = topk(prediction_a, labels_a, 5)
    it.set_postfix(acc_a=u'{:.3f}±{:.3f}'.format(
        np.array(acc_list).sum() * 100.0 / float(tt + 1),
        np.array(acc_list).std() / np.sqrt(float(tt + 1)) * 100.0))
  results_dict = {
      'acc': acc_list.mean(),
      'acc_se': stderr(acc_list),
      'acc_top5': acc_top5_list.mean(),
      'acc_top5_se': stderr(acc_top5_list)
  }
  return results_dict


def evaluate_b(sess, model, task_it, num_steps, old_and_new):
  """Evaluate the model on task A."""
  acc_list = np.zeros([num_steps])
  acc_top5_list = np.zeros([num_steps])
  if old_and_new:
    acc_list_old = np.zeros([num_steps])
    acc_list_old2 = np.zeros([num_steps])
    acc_list_new = np.zeros([num_steps])
    acc_list_new2 = np.zeros([num_steps])
    acc_top5_list_old = np.zeros([num_steps])
    acc_top5_list_old2 = np.zeros([num_steps])
    acc_top5_list_new = np.zeros([num_steps])
    acc_top5_list_new2 = np.zeros([num_steps])
  it = tqdm(six.moves.xrange(num_steps), ncols=0)
  for tt in it:
    task_data = task_it.next()
    prediction_b, labels_b = model.eval_step_b(sess, task_data)
    acc_list[tt] = top1(prediction_b, labels_b)
    acc_top5_list[tt] = topk(prediction_b, labels_b, 5)

    if old_and_new:
      old_idx = labels_b < model.num_classes_a
      new_idx = labels_b >= model.num_classes_a
      prediction_b_new = prediction_b[:, model.num_classes_a:]
      prediction_b_old = prediction_b[:, :model.num_classes_a]
      labels_b_new = labels_b - model.num_classes_a
      acc_list_old[tt] = top1(prediction_b[old_idx], labels_b[old_idx])
      acc_list_old2[tt] = top1(prediction_b_old[old_idx], labels_b[old_idx])
      acc_list_new[tt] = top1(prediction_b[new_idx], labels_b[new_idx])
      acc_list_new2[tt] = top1(prediction_b_new[new_idx],
                               labels_b_new[new_idx])
      acc_top5_list_old[tt] = topk(prediction_b[old_idx], labels_b[old_idx], 5)
      acc_top5_list_old2[tt] = topk(prediction_b_old[old_idx],
                                    labels_b[old_idx], 5)
      acc_top5_list_new[tt] = topk(prediction_b[new_idx], labels_b[new_idx], 5)
      acc_top5_list_new2[tt] = topk(prediction_b_new[new_idx],
                                    labels_b_new[new_idx], 5)

      it.set_postfix(
          acc_b=u'{:.3f}±{:.3f}'.format(acc_list[:tt + 1].mean() * 100.0,
                                        stderr(acc_list[:tt + 1]) * 100.0),
          acc_b_old=u'{:.3f}±{:.3f}'.format(
              acc_list_old[:tt + 1].mean() * 100.0,
              stderr(acc_list_old[:tt + 1]) * 100.0),
          acc_b_new=u'{:.3f}±{:.3f}'.format(
              acc_list_new[:tt + 1].mean() * 100.0,
              stderr(acc_list_new[:tt + 1]) * 100.0),
          acc_b_new2=u'{:.3f}±{:.3f}'.format(
              acc_list_new2[:tt + 1].mean() * 100.0,
              stderr(acc_list_new2[:tt + 1]) * 100.0))
    else:
      it.set_postfix(
          acc_b=u'{:.3f}±{:.3f}'.format(acc_list[:tt + 1].mean() * 100.0,
                                        stderr(acc_list[:tt + 1]) * 100.0))

  results_dict = {
      'acc': acc_list.mean(),
      'acc_se': stderr(acc_list),
      'acc_top5': acc_top5_list.mean(),
      'acc_top5_se': stderr(acc_top5_list)
  }

  if old_and_new:
    results_dict['acc_old'] = acc_list_old.mean()
    results_dict['acc_old_se'] = stderr(acc_list_old)
    results_dict['acc_old2'] = acc_list_old2.mean()
    results_dict['acc_old2_se'] = stderr(acc_list_old2)
    results_dict['acc_new'] = acc_list_new.mean()
    results_dict['acc_new_se'] = stderr(acc_list_new)
    results_dict['acc_new2'] = acc_list_new2.mean()
    results_dict['acc_new2_se'] = stderr(acc_list_new2)
    results_dict['acc_top5_old'] = acc_top5_list_old.mean()
    results_dict['acc_top5_old_se'] = stderr(acc_top5_list_old)
    results_dict['acc_top5_old2'] = acc_top5_list_old2.mean()
    results_dict['acc_top5_old2_se'] = stderr(acc_top5_list_old2)
    results_dict['acc_top5_new'] = acc_top5_list_new.mean()
    results_dict['acc_top5_new_se'] = stderr(acc_top5_list_new)
    results_dict['acc_top5_new2'] = acc_top5_list_new2.mean()
    results_dict['acc_top5_new2_se'] = stderr(acc_top5_list_new2)
  return results_dict


def preprocess_old_and_new(num_classes_a, task_a_it, task_b_it):
  """Combining two iterators into a single iterator. A regular B few-shot"""

  class Iterator():

    def next(self):
      # Increment class indices.
      task_b_data = task_b_it.next()
      task_b_data._y_train += num_classes_a
      task_b_data._y_test += num_classes_a
      task_a_data_test = task_a_it.next(forbid=task_b_data.y_sel)
      x_test_a, y_test_a = task_a_data_test

      # Combine old and new in the validation.
      task_b_data._x_test = np.concatenate([x_test_a, task_b_data.x_test],
                                           axis=0)
      task_b_data._y_test = np.concatenate([y_test_a, task_b_data.y_test],
                                           axis=0)
      return task_b_data

    def stop(self):
      task_b_it.stop()

  return Iterator()


def traineval(sess, model, modelv, logger, step, task_a_it, task_a_eval_it,
              task_b_it, task_b_eval_it, num_eval, old_and_new):
  """Run eval during training."""
  results_train_a = evaluate_a(sess, model, task_a_it, num_eval)
  results_eval_a = evaluate_a(sess, modelv, task_a_eval_it, num_eval)
  results_train_b = evaluate_b(sess, model, task_b_it, num_eval, old_and_new)
  results_eval_b = evaluate_b(sess, modelv, task_b_eval_it, num_eval,
                              old_and_new)
  train_acc_a = results_train_a['acc']
  train_acc_b = results_train_b['acc']
  val_acc_a = results_eval_a['acc']
  val_acc_b = results_eval_b['acc']

  if logger is not None:
    step_ = step + 1
    logger.log('train acc_a', step_, train_acc_a)
    logger.log('train acc_b', step_, train_acc_b)
    logger.log('val acc_a', step_, val_acc_a)
    logger.log('val acc_b', step_, val_acc_b)
    logger.log('learn_rate', step_, sess.run(model.learn_rate))

    if old_and_new:
      logger.log('train acc_old', step_, results_train_b['acc_old'])
      logger.log('train acc_new', step_, results_train_b['acc_new'])
      logger.log('train acc_new2', step_, results_train_b['acc_new2'])
      logger.log('val acc_old', step_, results_eval_b['acc_old'])
      logger.log('val acc_new', step_, results_eval_b['acc_new'])
      logger.log('val acc_new2', step_, results_eval_b['acc_new2'])
    logger.flush()
    print()
  return results_train_a, results_eval_a, results_train_b, results_eval_b


def train(sess,
          model,
          task_a_it,
          task_a_eval_it,
          task_b_it,
          task_b_eval_it,
          logger=None,
          saver=None,
          modelv=None,
          pretrain_a=0):
  """Train the model."""
  N = model.config.optimizer_config.max_train_steps
  it = six.moves.xrange(N)
  it = tqdm(it, ncols=0)
  train_acc_a = 0.0
  train_acc_b = 0.0
  val_acc_a = 0.0
  val_acc_b = 0.0
  num_eval = FLAGS.nepisode
  cost_a_steps = 0
  cost_a = None
  cost_b = None
  best_val_acc_b = None
  best_ckpt = None
  config = model.config.train_config
  old_and_new = model.config.transfer_config.old_and_new

  if modelv is None:
    modelv = model

  # ------------------------------------------------------------------------
  # Training loop
  for ii in it:

    # Run training one step.
    if ii < pretrain_a:
      # Pretrain on Task A.
      cost_a = model.train_step_a(sess, task_a_it.next())
    else:
      cost_a, cost_b, cost_b_v = model.train_step(sess, task_a_it.next(),
                                                  task_b_it.next())

    # Run evaluation.
    if (ii + 1) % config.steps_per_val == 0 or ii == 0:
      ra, rav, rb, rbv = traineval(sess, model, modelv, logger, ii, task_a_it,
                                   task_a_eval_it, task_b_it, task_b_eval_it,
                                   num_eval, old_and_new)

      # Save the best checkpoint so far.
      if best_val_acc_b is None or best_val_acc_b < rbv['acc']:
        best_val_acc_b = rbv['acc']
      if best_val_acc_b == rbv['acc'] and saver is not None:
        saver.save(sess, 'best-{}'.format(ii + 1))
        best_ckpt = 'best-{}'.format(ii + 1)
      # pass

    # Write logs.
    if ((ii + 1) % config.steps_per_log == 0 or ii == 0):
      if logger is not None:
        if cost_a is not None:
          logger.log('cost_a', ii + 1, cost_a)
        if cost_b is not None:
          logger.log('cost_b', ii + 1, cost_b)
          logger.log('cost_b_v', ii + 1, cost_b_v)
        logger.flush()

      # Update progress bar.
      post_fix_dict = {}
      if cost_a is not None:
        post_fix_dict['cost_a'] = '{:.3e}'.format(cost_a)
      if cost_b is not None:
        post_fix_dict['cost_b'] = '{:.3e}'.format(cost_b)
        post_fix_dict['cost_b_v'] = '{:.3e}'.format(cost_b_v)
      if old_and_new:
        post_fix_dict['vacc_o'] = '{:.3f}'.format(rbv['acc_old'] * 100.0)
        post_fix_dict['vacc_o2'] = '{:.3f}'.format(rbv['acc_old2'] * 100.0)
        post_fix_dict['vacc_n'] = '{:.3f}'.format(rbv['acc_new'] * 100.0)
        post_fix_dict['vacc_n2'] = '{:.3f}'.format(rbv['acc_new2'] * 100.0)
      post_fix_dict['vacc_a'] = '{:.3f}'.format(rav['acc'] * 100.0)
      post_fix_dict['vacc_b'] = '{:.3f}'.format(rbv['acc'] * 100.0)
      post_fix_dict['lr'] = '{:.3e}'.format(sess.run(model.learn_rate))
      it.set_postfix(**post_fix_dict)


def get_config(config_file):
  """Reads configuration."""
  config = ExperimentConfig()
  Merge(open(config_file).read(), config)
  return config


def get_model(config,
              num_classes_a,
              nclasses_train,
              nclasses_val,
              nclasses_test,
              is_eval=False):
  """Builds model."""
  assert config.backbone_class == 'resnet_backbone', 'Only support ResNet'
  bb_config = config.resnet_config

  # ------------------------------------------------------------------------
  # Placeholders
  x_a = tf.placeholder(
      tf.float32,
      [None, bb_config.height, bb_config.width, bb_config.num_channel],
      name='x_a')
  y_a = tf.placeholder(tf.int64, [None], name='y_a')
  x_b = tf.placeholder(
      tf.float32,
      [None, bb_config.height, bb_config.width, bb_config.num_channel],
      name='x_b')
  y_b = tf.placeholder(tf.int64, [None], name='y_b')
  y_sel = tf.placeholder(tf.int64, [None], name='y_sel')

  x_b_v = tf.placeholder(
      tf.float32,
      [None, bb_config.height, bb_config.width, bb_config.num_channel],
      name='x_b_v')
  y_b_v = tf.placeholder(tf.int64, [None], name='y_b_v')

  # ------------------------------------------------------------------------
  # Model classes
  if config.model_class == "multitask":
    model_class = MultiTaskModel
  elif config.model_class == "imprint":
    model_class = ImprintModel
  elif config.model_class == "attractor":
    model_class = AttractorModel
  elif config.model_class == "attractor-bptt":
    model_class = AttractorModelBPTT
  else:
    raise ValueError("Unknown model")

  ext_wts = None

  # ------------------------------------------------------------------------
  # Build model for training
  with tf.name_scope('Train'):
    with tf.variable_scope('Model'):
      model = model_class(
          config,
          x_a,
          y_a,
          x_b,
          y_b,
          x_b_v,
          y_b_v,
          num_classes_a,
          nclasses_train,
          is_training=True,
          ext_wts=ext_wts,
          y_sel=y_sel)
  [log.info(v.name) for v in tf.global_variables()]

  # ------------------------------------------------------------------------
  # Build model for eval
  with tf.name_scope('Val'):
    reuse_eval = tf.AUTO_REUSE
    with tf.variable_scope('Model', reuse=reuse_eval):
      modelv = model_class(
          config,
          x_a,
          y_a,
          x_b,
          y_b,
          x_b_v,
          y_b_v,
          num_classes_a,
          nclasses_val,
          is_training=False,
          ext_wts=ext_wts,
          y_sel=y_sel)

  if nclasses_val == nclasses_test:
    modelt = modelv
  else:
    with tf.name_scope('Test'):
      with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        modelt = model_class(
            config,
            x_a,
            y_a,
            x_b,
            y_b,
            x_b_v,
            y_b_v,
            num_classes_a,
            nclasses_test,
            is_training=False,
            ext_wts=ext_wts,
            y_sel=y_sel)
  return {'train': model, 'val': modelv, 'test': modelt}


def get_datasets(dataset, metadata, nshot, num_test, batch_size, num_gpu,
                 nclasses_a, nclasses_train, nclasses_val, nclasses_test,
                 old_and_new, seed, is_eval):
  """Builds datasets"""
  # ------------------------------------------------------------------------
  # Datasets
  train_dataset_a = get_dataset(
      dataset,
      metadata['trainsplit_a_train'],
      nclasses_train,
      nshot,
      label_ratio=metadata['label_ratio'],
      num_test=num_test // num_gpu,
      aug_90=False,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed,
      image_split_file=metadata['image_split_file_a_train'],
      nclasses=nclasses_a)
  if metadata['trainsplit_b'] == metadata['trainsplit_a_train']:
    train_dataset_b = train_dataset_a
  else:
    train_dataset_b = get_dataset(
        dataset,
        metadata['trainsplit_b'],
        nclasses_train,
        nshot,
        label_ratio=1.0,
        num_test=num_test // num_gpu,
        aug_90=False,
        num_unlabel=0,
        shuffle_episode=False,
        seed=seed,
        image_split_file=metadata['image_split_file_b'])
  trainval_dataset_a = get_dataset(
      dataset,
      metadata['trainsplit_a_val'],
      nclasses_train,
      nshot,
      label_ratio=metadata['label_ratio'],
      num_test=num_test // num_gpu,
      aug_90=False,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed,
      image_split_file=metadata['image_split_file_a_val'],
      nclasses=nclasses_a)
  traintest_dataset_a = get_dataset(
      dataset,
      metadata['trainsplit_a_test'],
      nclasses_train,
      nshot,
      label_ratio=metadata['label_ratio'],
      num_test=num_test // num_gpu,
      aug_90=False,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed,
      image_split_file=metadata['image_split_file_a_test'],
      nclasses=nclasses_a)
  val_dataset = get_dataset(
      dataset,
      'val',
      nclasses_val,
      nshot,
      label_ratio=1.0,
      num_test=num_test // num_gpu,
      aug_90=False,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed)
  test_dataset = get_dataset(
      dataset,
      "test",
      nclasses_test,
      nshot,
      num_test=num_test // num_gpu,
      label_ratio=1.0,
      aug_90=False,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed)

  # ------------------------------------------------------------------------
  # Task A iterators
  task_a_iter = get_iter(
      train_dataset_a.get_size(),
      train_dataset_a.get_batch_idx,
      batch_size // num_gpu,
      cycle=True,
      shuffle=True,
      max_queue_size=10,
      num_threads=2,
      seed=seed)
  task_a_val_iter = get_iter(
      trainval_dataset_a.get_size(),
      trainval_dataset_a.get_batch_idx,
      batch_size // num_gpu,
      cycle=True,
      shuffle=True,
      max_queue_size=10,
      num_threads=2,
      seed=seed)
  task_a_test_iter = get_iter(
      traintest_dataset_a.get_size(),
      traintest_dataset_a.get_batch_idx,
      batch_size // num_gpu,
      cycle=True,
      shuffle=True,
      max_queue_size=10,
      num_threads=2,
      seed=seed)

  # ------------------------------------------------------------------------
  # Task B iterators
  task_b_iter = get_concurrent_iterator(
      train_dataset_b, max_queue_size=2, num_threads=1)
  task_b_val_iter = get_concurrent_iterator(
      val_dataset, max_queue_size=2, num_threads=1)
  task_b_test_iter = get_concurrent_iterator(
      test_dataset, max_queue_size=2, num_threads=1)

  # ------------------------------------------------------------------------
  # Task B iterators for old and new (wrapper)
  if old_and_new:
    task_a_iter_old = get_iter(
        train_dataset_a.get_size(),
        train_dataset_a.get_batch_idx,
        num_test * nclasses_train // num_gpu,
        cycle=True,
        shuffle=True,
        max_queue_size=10,
        num_threads=1,
        seed=seed + 1)
    task_a_val_iter_old = get_iter(
        trainval_dataset_a.get_size(),
        trainval_dataset_a.get_batch_idx,
        num_test * nclasses_val // num_gpu,
        cycle=True,
        shuffle=True,
        max_queue_size=10,
        num_threads=1,
        seed=seed + 1)
    task_a_test_iter_old = get_iter(
        traintest_dataset_a.get_size(),
        traintest_dataset_a.get_batch_idx,
        num_test * nclasses_test // num_gpu,
        cycle=True,
        shuffle=True,
        max_queue_size=10,
        num_threads=1,
        seed=seed + 1)

    if nclasses_a == -1:
      num_classes_a = metadata['num_classes_a']
    else:
      num_classes_a = nclasses_a
    task_b_iter = preprocess_old_and_new(num_classes_a, task_a_iter_old,
                                         task_b_iter)
    task_b_val_iter = preprocess_old_and_new(
        num_classes_a, task_a_val_iter_old, task_b_val_iter)
    task_b_test_iter = preprocess_old_and_new(
        num_classes_a, task_a_test_iter_old, task_b_test_iter)

  results = {}
  results['a_train'] = task_a_iter
  results['a_val'] = task_a_val_iter
  results['a_test'] = task_a_test_iter
  results['b_train'] = task_b_iter
  results['b_val'] = task_b_val_iter
  results['b_test'] = task_b_test_iter

  return results


def get_restore_saver(retest=False, cosine_a=False, reinit_tau=False):
  """Gets restore saver."""
  var_list = tf.global_variables()
  if not retest:
    # Testing only.
    forbid_list = [
        'global_step', 'Adam', 'w_p1', 'w_p2', 'w_q', 'k_b', 'tau_q', 'tau_b',
        'Momentum', 'w_class_b', 'b_class_b', 'ft_step', 'transfer_loss',
        'new_loss', 'Optimizer', 'grads_b', 'beta1_power', 'beta2_power'
    ]
    if reinit_tau:
      forbid_list += ['tau']
  else:
    # For training second stage.
    forbid_list = [
        'ft_step', 'w_class_b', 'b_class_b', 'Adam', 'Optimizer', 'grads_b',
        'Momentum', 'beta1_power', 'beta2_power'
    ]
  log.info('Forbid restore list: {}'.format(forbid_list))
  condition = lambda x: not any([forbid in x.name for forbid in forbid_list])
  var_list = list(filter(condition, var_list))
  var_keys = [v.name.split(':')[0] for v in var_list]
  var_dict = dict(zip(var_keys, var_list))
  restore_saver = tf.train.Saver(var_dict)
  return restore_saver


def restore_model(sess,
                  model,
                  modelv,
                  restore_saver,
                  is_eval=False,
                  pretrain=None):
  """Restore model from checkpoint."""
  if pretrain is not None:
    log_folder_restore = pretrain
    log.info('Restore from {}'.format(log_folder_restore))
    ckpt = tf.train.latest_checkpoint(log_folder_restore)
    log.info('Checkpoint: {}'.format(ckpt))
    if is_eval:
      modelv.initialize(sess)
    else:
      model.initialize(sess)
    restore_saver.restore(sess, ckpt)
  else:
    modelv.initialize(sess)


def _log_line(f, name, acc, se):
  """Log one line.

  Args:
    f: File object.
    name: String. Name of the metric.
    acc: Float. Accuracy.
    se: Float. Standard error.
  """
  msg1 = '\t'.join([
      '{:15s}'.format(name), '{:.3f}'.format(acc * 100.0),
      '({:.3f})'.format(se * 100.0)
  ])
  msg2 = '\t'.join(
      [name, '{:.3f}'.format(acc * 100.0), '{:.3f}'.format(se * 100.0)])
  log.info(msg1)
  f.write(msg2 + '\n')


def final_log(log_folder, results, filename='results.tsv', old_and_new=False):
  """Log final performance numbers."""
  log_filename = os.path.join(log_folder, 'results.tsv')
  f = open(log_filename, 'a')
  f.write('\t'.join(['Name\t', 'Value', 'SE']) + '\n')
  formatname = lambda x: x.replace('acc', '').replace('_', ' ').title()

  for n in ['train_a', 'val_a', 'test_a']:
    if n in results:
      _results = results[n]
      _log_line(f, formatname(n) + ' Acc', _results['acc'], _results['acc_se'])

  for n in ['train_b', 'val_b', 'test_b']:
    if n in results:
      _results = results[n]
      if 'acc' in _results:
        _log_line(f,
                  formatname(n) + ' Acc', _results['acc'], _results['acc_se'])
      if 'acc_top5' in _results:
        _log_line(f,
                  formatname(n) + ' Top5 Acc', _results['acc_top5'],
                  _results['acc_top5_se'])
      for m in [
          'acc_new', 'acc_new2', 'acc_old', 'acc_old2', 'acc_top5_new',
          'acc_top5_new2', 'acc_top5_old', 'acc_top5_old2'
      ]:
        if m in _results:
          _log_line(f,
                    formatname(n) + ' ' + formatname(m), _results[m],
                    _results[m + '_se'])

  f.close()


def main():
  # ------------------------------------------------------------------------
  # Flags
  nshot = FLAGS.nshot
  dataset = FLAGS.dataset
  nclasses_train = FLAGS.nclasses_b
  nclasses_val = FLAGS.nclasses_b
  nclasses_test = FLAGS.nclasses_b
  nclasses_a = FLAGS.nclasses_a
  num_test = FLAGS.ntest
  is_eval = FLAGS.eval
  nepisode = FLAGS.nepisode
  nepisode_final = FLAGS.nepisode_final
  run_val = FLAGS.val
  run_test = FLAGS.test
  pretrain = FLAGS.pretrain
  retest = FLAGS.retest
  tag = FLAGS.tag

  # ------------------------------------------------------------------------
  # Configuration
  config = get_config(FLAGS.config)
  opt_config = config.optimizer_config
  old_and_new = config.transfer_config.old_and_new

  # ------------------------------------------------------------------------
  # Log folder
  assert tag is not None, 'Please add a name for the experiment'
  log_folder = os.path.join(FLAGS.results, dataset, tag)
  log.info('Experiment ID {}'.format(tag))
  if os.path.exists(log_folder) and not FLAGS.eval:
    assert False, 'Folder {} exists. Pick another tag.'.format(log_folder)

  # ------------------------------------------------------------------------
  # Model
  metadata = get_metadata(dataset)
  if nclasses_a == -1:
    num_classes_a = metadata['num_classes_a']
  else:
    num_classes_a = nclasses_a
    log.info('Use total number of classes = {}'.format(num_classes_a))
  with log.verbose_level(2):
    model_dict = get_model(
        config,
        num_classes_a,
        nclasses_train,
        nclasses_val,
        nclasses_test,
        is_eval=is_eval)
    model = model_dict['train']
    modelv = model_dict['val']
    modelt = model_dict['test']

  # ------------------------------------------------------------------------
  # Dataset
  seed = 0

  with log.verbose_level(2):
    data = get_datasets(dataset, metadata, nshot, num_test,
                        opt_config.batch_size, opt_config.num_gpu, nclasses_a,
                        nclasses_train, nclasses_val, nclasses_test,
                        old_and_new, seed, is_eval)

  # ------------------------------------------------------------------------
  # Save configurations
  save_config(config, log_folder)

  # ------------------------------------------------------------------------
  # Log outputs
  restore_saver = get_restore_saver(
      retest=retest,
      cosine_a=modelv.config.protonet_config.cosine_a,
      reinit_tau=modelv.config.protonet_config.reinit_tau)
  logger = get_exp_logger(log_folder)
  saver = get_saver(log_folder)

  # ------------------------------------------------------------------------
  # Create a TensorFlow session
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess = tf.Session(config=sess_config)

  # ------------------------------------------------------------------------
  # Initialize model
  restore_model(
      sess, model, modelv, restore_saver, is_eval=is_eval, pretrain=pretrain)

  # ------------------------------------------------------------------------
  # Training
  if not is_eval:
    train(
        sess,
        model,
        data['a_train'],
        data['a_val'],
        data['b_train'],
        data['b_val'],
        logger=logger,
        saver=saver,
        modelv=modelv,
        pretrain_a=model.config.train_config.pretrain_a_steps)
    saver.save(sess, 'last')

  # ------------------------------------------------------------------------
  # Testing
  log.info('Experiment ID {}'.format(tag))
  results = {}
  nepisode_a = nepisode // 5
  results['train_a'] = evaluate_a(sess, model, data['a_train'], nepisode_a)
  results['val_a'] = evaluate_a(sess, modelv, data['a_val'], nepisode_a)
  results['test_a'] = evaluate_a(sess, modelv, data['a_test'], nepisode_a)
  results['train_b'] = evaluate_b(sess, model, data['b_train'], nepisode_final,
                                  old_and_new)
  if run_val:
    results['val_b'] = evaluate_b(sess, modelv, data['b_val'], nepisode_final,
                                  old_and_new)
  if run_test:
    results['test_b'] = evaluate_b(sess, modelt, data['b_test'],
                                   nepisode_final, old_and_new)
  final_log(log_folder, results, old_and_new=old_and_new)


if __name__ == '__main__':
  main()
