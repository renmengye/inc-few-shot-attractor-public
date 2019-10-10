"""Runs a baseline for prototype networks for incremental few-shot learning.

Author: Mengye Ren (mren@cs.toronto.edu)

See run_exp.py for usage.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import six
import tensorflow as tf

from tqdm import tqdm

from fewshot.utils import logger
from run_exp import (get_config, get_restore_saver, get_datasets, get_model,
                     save_config, get_exp_logger, get_saver, restore_model,
                     final_log)
from train_lib import get_metadata

log = logger.get()
FLAGS = tf.flags.FLAGS


def calculate_protos(sess, model, num_classes_a, task_a_it, num_steps):
  """Calculates the prototypes of the entire training set."""
  prototypes = []
  for idx in six.moves.xrange(num_classes_a):
    prototypes.append([])
  for step in six.moves.xrange(num_steps):
    x, y = task_a_it.next()
    h = sess.run(model.h_a, feed_dict={model.inputs: x})
    for jj, idx in enumerate(y):
      prototypes[idx].append(h[jj])
  for idx in six.moves.xrange(num_classes_a):
    prototypes[idx] = np.array(prototypes[idx]).mean(axis=0)
  return np.array(prototypes)


def calculate_episode_protos(sess, model, num_classes_a, nway, episode,
                             old_and_new):
  """Caluclates the prototypes of a single episode."""
  prototypes = []
  for idx in six.moves.xrange(nway):
    prototypes.append([])
  h = sess.run(model.h_a, feed_dict={model.inputs: episode.x_train})
  for idx in six.moves.xrange(episode.x_train.shape[0]):
    if old_and_new:
      prototypes[episode.y_train[idx] - num_classes_a].append(h[idx])
    else:
      prototypes[episode.y_train[idx]].append(h[idx])
  for idx in six.moves.xrange(nway):
    prototypes[idx] = np.array(prototypes[idx]).mean(axis=0)
  return np.array(prototypes)


def cosine(h, protos):
  """Cosine similarity."""
  proto_t = protos.T
  result = np.dot(h, proto_t) / np.sqrt(np.sum(
      h**2, axis=1, keepdims=True)) / np.sqrt(
          np.sum(proto_t**2, axis=0, keepdims=True))
  return result


def euclidean(h, protos):
  """Euclidean similarity."""
  h_ = np.expand_dims(h, 1)
  protos_ = np.expand_dims(protos, 0)
  return -np.sum((h_ - protos_)**2, axis=2)


def dot(h, protos):
  """Dot product."""
  return np.dot(h, protos.T)


def evaluate_b(sess,
               model,
               task_it,
               num_steps,
               num_classes_a,
               num_classes_b,
               prototypes_a=None,
               old_and_new=False,
               similarity='euclidean'):
  """Evaluate the model on task A."""
  acc_list = np.zeros([num_steps])
  if old_and_new:
    acc_list_old = np.zeros([num_steps])
    acc_list_new = np.zeros([num_steps])
    acc_list_old2 = np.zeros([num_steps])
    acc_list_new2 = np.zeros([num_steps])
  it = tqdm(six.moves.xrange(num_steps), ncols=0)

  for tt in it:
    task_data = task_it.next()
    prototypes_b = calculate_episode_protos(
        sess, model, num_classes_a, num_classes_b, task_data, old_and_new)
    if old_and_new:
      all_prototypes = np.concatenate([prototypes_a, prototypes_b])
    else:
      all_prototypes = prototypes_b
    h_test = sess.run(model.h_a, feed_dict={model.inputs: task_data.x_test})

    if similarity == 'cosine':
      logits = cosine(h_test, all_prototypes)
    elif similarity == 'euclidean':
      logits = euclidean(h_test, all_prototypes)
    elif similarity == 'dot':
      logits = dot(h_test, all_prototypes)
    else:
      raise ValueError('Unknown similarity function')

    correct = np.equal(np.argmax(logits, axis=1),
                       task_data.y_test).astype(np.float32)
    _acc = correct.mean()
    acc_list[tt] = _acc
    if old_and_new:
      is_new = task_data.y_test >= num_classes_a
      is_old = np.logical_not(is_new)
      _acc_old = correct[is_old].mean()
      _acc_new = correct[is_new].mean()
      correct_new = np.equal(
          np.argmax(logits[is_new, num_classes_a:], axis=1),
          task_data.y_test[is_new] - num_classes_a).astype(np.float32)
      _acc_new2 = correct_new.mean()
      correct_old = np.equal(
          np.argmax(logits[is_old, :num_classes_a], axis=1),
          task_data.y_test[is_old]).astype(np.float32)
      _acc_old2 = correct_old.mean()
      acc_list_old[tt] = _acc_old
      acc_list_new[tt] = _acc_new
      acc_list_new2[tt] = _acc_new2
      acc_list_old2[tt] = _acc_old2
      it.set_postfix(
          acc_b=u'{:.3f}±{:.3f}'.format(
              np.array(acc_list).sum() * 100.0 / float(tt + 1),
              np.array(acc_list).std() / np.sqrt(float(tt + 1)) * 100.0),
          acc_b_old=u'{:.3f}±{:.3f}'.format(
              np.array(acc_list_old).sum() * 100.0 / float(tt + 1),
              np.array(acc_list_old).std() / np.sqrt(float(tt + 1)) * 100.0),
          acc_b_old2=u'{:.3f}±{:.3f}'.format(
              np.array(acc_list_old2).sum() * 100.0 / float(tt + 1),
              np.array(acc_list_old2).std() / np.sqrt(float(tt + 1)) * 100.0),
          acc_b_new=u'{:.3f}±{:.3f}'.format(
              np.array(acc_list_new).sum() * 100.0 / float(tt + 1),
              np.array(acc_list_new).std() / np.sqrt(float(tt + 1)) * 100.0),
          acc_b_new2=u'{:.3f}±{:.3f}'.format(
              np.array(acc_list_new2).sum() * 100.0 / float(tt + 1),
              np.array(acc_list_new2).std() / np.sqrt(float(tt + 1)) * 100.0))
    else:
      it.set_postfix(acc_b=u'{:.3f}±{:.3f}'.format(
          np.array(acc_list).sum() * 100.0 / float(tt + 1),
          np.array(acc_list).std() / np.sqrt(float(tt + 1)) * 100.0))
  results_dict = {
      'acc': acc_list.mean(),
      'acc_se': acc_list.std() / np.sqrt(float(acc_list.size))
  }

  if old_and_new:
    results_dict['acc_old'] = acc_list_old.mean()
    results_dict['acc_old_se'] = acc_list_old.std() / np.sqrt(
        float(acc_list_old.size))
    results_dict['acc_old2'] = acc_list_old2.mean()
    results_dict['acc_old2_se'] = acc_list_old2.std() / np.sqrt(
        float(acc_list_old2.size))
    results_dict['acc_new'] = acc_list_new.mean()
    results_dict['acc_new_se'] = acc_list_new.std() / np.sqrt(
        float(acc_list_new.size))
    results_dict['acc_new2'] = acc_list_new2.mean()
    results_dict['acc_new2_se'] = acc_list_new2.std() / np.sqrt(
        float(acc_list_new2.size))
  return results_dict


def main():
  # ------------------------------------------------------------------------
  # Flags
  nshot = FLAGS.nshot
  dataset = FLAGS.dataset
  nclasses_train = FLAGS.nclasses_b
  nclasses_val = FLAGS.nclasses_b
  nclasses_test = FLAGS.nclasses_b
  num_test = FLAGS.ntest
  is_eval = FLAGS.eval
  nepisode = FLAGS.nepisode
  run_test = FLAGS.test
  pretrain = FLAGS.pretrain
  retest = FLAGS.retest
  tag = FLAGS.tag

  # ------------------------------------------------------------------------
  # Configuration
  config = get_config(FLAGS.config)
  opt_config = config.optimizer_config
  old_and_new = config.transfer_config.old_and_new
  similarity = config.protonet_config.similarity

  # ------------------------------------------------------------------------
  # Log folder
  assert tag is not None, 'Please add a name for the experiment'
  log_folder = os.path.join(FLAGS.results, dataset, 'n{}w{}'.format(
      nshot, nclasses_val), tag)
  log.info('Experiment ID {}'.format(tag))
  if not os.path.exists(log_folder):
    os.makedirs(log_folder)
  elif not is_eval:
    assert False, 'Folder {} exists. Pick another tag.'.format(log_folder)

  # ------------------------------------------------------------------------
  # Model
  metadata = get_metadata(dataset)
  with log.verbose_level(2):
    model_dict = get_model(
        config,
        metadata['num_classes_a'],
        nclasses_train,
        nclasses_val,
        nclasses_test,
        is_eval=is_eval)
    model = model_dict['val']
    modelv = model_dict['val']

  # ------------------------------------------------------------------------
  # Dataset
  seed = 0

  with log.verbose_level(2):
    data = get_datasets(dataset, metadata, nshot, num_test,
                        opt_config.batch_size, opt_config.num_gpu,
                        metadata['num_classes_a'], nclasses_train, nclasses_val,
                        nclasses_test, old_and_new, seed, True)

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
  # Calculate prototypes A.
  if old_and_new:
    prototypes_a = calculate_protos(sess, model, model.num_classes_a,
                                    data['a_train'], nepisode)
  else:
    prototypes_a = None

  # ------------------------------------------------------------------------
  # Run on val set.
  results = {}
  results['val_b'] = evaluate_b(
      sess,
      model,
      data['b_val'],
      nepisode,
      model.num_classes_a,
      nclasses_val,
      prototypes_a=prototypes_a,
      old_and_new=old_and_new,
      similarity=similarity)

  # ------------------------------------------------------------------------
  # Run on test set.
  if run_test:
    results['test_b'] = evaluate_b(
        sess,
        model,
        data['b_test'],
        nepisode,
        model.num_classes_a,
        nclasses_val,
        prototypes_a=prototypes_a,
        old_and_new=old_and_new,
        similarity=similarity)

  # ------------------------------------------------------------------------
  # Log results.
  final_log(log_folder, results, old_and_new=old_and_new)


if __name__ == '__main__':
  main()
