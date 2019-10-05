"""Dataset API for miniImageNet.

Authors:
Sachin Ravi (sachinr@cs.princeton.edu)
Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import cv2
import numpy as np
import os
import pickle as pkl
import tensorflow as tf

from collections import namedtuple

from fewshot.data.data_factory import RegisterDataset
from fewshot.data.episode import Episode
from fewshot.utils import logger

AL_Instance = namedtuple('AL_Instance',
                         'n_class, n_distractor, k_train, k_test, k_unlbl')
DatasetT = namedtuple('Dataset', 'data, labels')

# TODO: put in config file
N_IMAGES = 600
N_INPUT = 84
IMAGES_PATH = "images/"
CSV_FILES = {
    'train': 'fewshot/data/mini_imagenet_split/Ravi/train.csv',
    'train_a': 'fewshot/data/mini_imagenet_split/Ravi/train_a.csv',
    'train_b': 'fewshot/data/mini_imagenet_split/Ravi/train_b.csv',
    'val': 'fewshot/data/mini_imagenet_split/Ravi/val.csv',
    'test': 'fewshot/data/mini_imagenet_split/Ravi/test.csv'
}

# Fixed random seed to get same split of labeled vs unlabeled items for each
# class
FIXED_SEED = 22

log = logger.get()
FLAGS = tf.flags.FLAGS


@RegisterDataset("mini-imagenet")
class MiniImageNetDataset(object):

  def __init__(self,
               folder,
               split,
               nway=5,
               nshot=1,
               num_unlabel=5,
               num_distractor=5,
               num_test=15,
               split_def="",
               label_ratio=None,
               shuffle_episode=False,
               seed=FIXED_SEED,
               aug_90=False,
               use_unlabel=False,
               **kwargs):
    self._folder = folder
    log.info('Folder {}'.format(self._folder))
    self._split = split
    log.info('Split {}'.format(self._split))
    self._seed = seed
    self._num_distractor = 0
    log.warning("Number of distractors in each episode: {}".format(
        self._num_distractor))
    no_label_ratio = label_ratio is None
    self._label_ratio = FLAGS.label_ratio if no_label_ratio else label_ratio
    log.info('Label ratio {}'.format(self._label_ratio))
    self.n_lbl = int(N_IMAGES * self._label_ratio)
    self._use_unlabel = False
    log.info("Num unlabel {}".format(num_unlabel))
    log.info("Num test {}".format(num_test))
    log.info("Num distractor {}".format(self._num_distractor))
    self.mean_pix = np.array(
        [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]])
    self.std_pix = np.array(
        [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])

    # define AL instance
    self.al_instance = AL_Instance(
        n_class=nway,
        n_distractor=self._num_distractor,
        k_train=nshot,
        k_test=num_test,
        k_unlbl=num_unlabel)
    self.n_input = N_INPUT

    self.images_path = os.path.join(self._folder, IMAGES_PATH)
    if not self._read_cache(split):
      self._write_cache(split, CSV_FILES[split])
    self.class_dict = self.split_label_unlabel(self.class_dict)
    self._num_classes = len(self.class_dict.keys())
    self._lbl_idx = []
    self._unlbl_idx = []
    self._cls_label = {}
    # cls_label = 0
    # print(self._idic)
    for kk in sorted(self.class_dict.keys()):
      cls_label = self.dic[kk]
      _nlbl = len(self.class_dict[kk]['lbl'])
      _nunlbl = len(self.class_dict[kk]['unlbl'])
      self._lbl_idx.extend(self.class_dict[kk]['lbl'])
      self._unlbl_idx.extend(self.class_dict[kk]['unlbl'])
      for idx in self.class_dict[kk]['lbl']:
        self._cls_label[idx] = cls_label
      for idx in self.class_dict[kk]['unlbl']:
        self._cls_label[idx] = cls_label
      # cls_label += 1
    self._lbl_idx = np.array(self._lbl_idx)
    self._unlbl_idx = np.array(self._unlbl_idx)
    self._num_lbl = len(self._lbl_idx)
    self._num_unlbl = len(self._unlbl_idx)
    log.info('Num label {}'.format(self._num_lbl))
    log.info('Num unlabel {}'.format(self._num_unlbl))

    with tf.device('/cpu:0'):
      self._rnd_process_plh, self._rnd_process = self.tf_preprocess(
          random_color=False)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=sess_config)

  def get_cache_path(self, split):
    """Gets cache file name."""
    cache_path = os.path.join(
        self._folder, "miniImageNet_category_split_{}.pickle".format(split))
    return cache_path

  def _read_cache(self, split):
    cache_path = self.get_cache_path(split)
    log.info('cache path {}'.format(cache_path))
    if os.path.exists(cache_path):
      try:
        with open(cache_path, "rb") as f:
          datafile = pkl.load(f, encoding='latin1')
          log.info('Keys {}'.format(datafile.keys()))
          data = datafile['data']
          labels = datafile['labels']
          dic = datafile['catname2label']
      except:
        with open(cache_path, "rb") as f:
          datafile = pkl.load(f)
          data = datafile['image_data']
          labels = datafile['class_dict']
          dic = datafile['catname2label']

      self.img_data = data
      keys = dic.keys()
      keys = list(sorted(keys))
      self.dic = dic  # key to idx.
      values = [dic[kk] for kk in keys]
      idic = dict(zip(values, keys))
      self.idic = idic  # idx to key.
      self.class_dict = {}
      for kk in keys:
        self.class_dict[kk] = []
      for ii, ll in enumerate(labels):
        self.class_dict[idic[ll]].append(ii)
      return True
    else:
      return False

  def _write_cache(self, split, csv_filename):
    cache_path = self.get_cache_path(split)
    img_data = []

    class_dict = {}
    i = 0
    with open(csv_filename) as csv_file:
      csv_reader = csv.reader(csv_file)
      for (image_filename, class_name) in csv_reader:
        if 'label' not in class_name:
          if class_name in class_dict:
            class_dict[class_name].append(i)
          else:
            class_dict[class_name] = [i]
          img_data.append(
              cv2.resize(
                  cv2.imread(self.images_path +
                             image_filename)[:, :, [2, 1, 0]],
                  (self.n_input, self.n_input)))
          i += 1

    self.img_data = np.stack(img_data)
    self.class_dict = class_dict
    data = {"image_data": self.img_data, "class_dict": self.class_dict}
    with open(cache_path, "wb") as f:
      pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

  def next(self, within_category=False, catcode=None):
    """
    (1) Pick random set of classes
    (2) Pick random partitioning into train, test, unlabeled
    """
    sel_classes = np.random.choice(
        range(len(self.class_dict.keys())),
        size=self.al_instance.n_class + self.al_instance.n_distractor,
        replace=False)
    # if len(self.class_dict.keys()) < 50:
    k_per_class = [
        None for i in range(self.al_instance.n_class +
                            self.al_instance.n_distractor)
    ]

    total_train = None
    total_test = None
    total_unlbl = None
    n_class = self.al_instance.n_class
    n_distractor = self.al_instance.n_distractor
    k_train = self.al_instance.k_train
    k_test = self.al_instance.k_test
    k_unlbl = self.al_instance.k_unlbl
    total_train = np.zeros([n_class * k_train, 84, 84, 3], dtype=np.float32)
    total_test = np.zeros([n_class * k_test, 84, 84, 3], dtype=np.float32)
    total_unlbl = np.zeros([(n_class + n_distractor) * k_unlbl, 84, 84, 3],
                           dtype=np.float32)
    total_train_label = np.zeros([n_class * k_train], dtype=np.int64)
    total_test_label = np.zeros([n_class * k_test], dtype=np.int64)
    total_unlbl_label = np.zeros([n_class * k_unlbl], dtype=np.int64)
    y_train_str = []
    y_test_str = []
    for idx, cl in enumerate(sel_classes[:self.al_instance.n_class]):
      train, test, unlbl = self._get_rand_partition(
          list(self.class_dict.keys())[cl], idx, k_per_class[idx])
      total_train[idx * k_train:(idx + 1) * k_train] = train
      total_test[idx * k_test:(idx + 1) * k_test] = test
      y_train_str.extend([cl] * k_train)
      y_test_str.extend([cl] * k_test)
      total_unlbl[idx * k_unlbl:(idx + 1) * k_unlbl] = unlbl
      total_train_label[idx * k_train:(idx + 1) * k_train] = idx
      total_test_label[idx * k_test:(idx + 1) * k_test] = idx
      total_unlbl_label[idx * k_unlbl:(idx + 1) * k_unlbl] = 1

    for idx, cl in enumerate(sel_classes[self.al_instance.n_class:]):
      unlbl = self._get_rand_partition(
          list(self.class_dict.keys())[cl], self.al_instance.n_class + idx,
          k_per_class[idx])
      total_unlbl[(idx + n_class) * k_unlbl:(idx + n_class + 1) *
                  k_unlbl] = unlbl

    if self._split == 'train_phase_train':
      y_sel = sel_classes[:self.al_instance.n_class]
    else:
      y_sel = None  # No need to forbid here.

    if self._split == 'train_phase_train':
      for jj in range(total_train.shape[0]):
        total_train[jj] = self._sess.run(
            self._rnd_process,
            feed_dict={self._rnd_process_plh: total_train[jj]})
      for jj in range(total_test.shape[0]):
        total_test[jj] = self._sess.run(
            self._rnd_process,
            feed_dict={self._rnd_process_plh: total_test[jj]})

    total_train = self.normalize(total_train)
    total_unlbl = self.normalize(total_unlbl)
    total_test = self.normalize(total_test)

    return Episode(
        x_train=total_train,
        y_train=total_train_label,
        x_test=total_test,
        y_test=total_test_label,
        x_unlabel=total_unlbl,
        y_unlabel=total_unlbl_label,
        y_train_str=y_train_str,
        y_test_str=y_test_str,
        y_sel=y_sel)

  def _read_csv(self, csv_filename):
    """ from csv file,
    store dictionary : class_names -> [name of class_images] """
    class_dict = {}
    with open(csv_filename) as csv_file:
      csv_reader = csv.reader(csv_file)
      for (image_filename, class_name) in csv_reader:
        if 'label' not in class_name:
          if class_name in class_dict:
            class_dict[class_name].append(image_filename)
          else:
            class_dict[class_name] = [image_filename]
    # convert dict: class_name ->
    # {
    #   'lbl': [name of labeled class images],
    #   'unlbl' : [name of unlabeled images]'
    # }
    new_class_dict = {}
    log.info('Seed {}'.format(self._seed))
    for class_name, image_list in class_dict.items():
      np.random.RandomState(self._seed).shuffle(image_list)
      new_class_dict[class_name] = {
          'lbl': image_list[0:self.n_lbl],
          'unlbl': image_list[self.n_lbl:]
      }

    return new_class_dict

  def split_label_unlabel(self, class_dict):
    splitfile = os.path.join(
        self._folder,
        "mini-imagenet-labelsplit-" + self._split + "-{:d}-{:d}.pkl".format(
            int(self._label_ratio * 100), self._seed))
    new_class_dict = {}
    for class_name, image_list in class_dict.items():
      np.random.RandomState(self._seed).shuffle(image_list)
      new_class_dict[class_name] = {
          'lbl': image_list[0:self.n_lbl],
          'unlbl': image_list[self.n_lbl:]
      }

    with open(splitfile, 'wb') as f:
      pkl.dump(new_class_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    return new_class_dict

  def _get_rand_partition(self, class_name, class_idx, k_unlbl=None):
    unlbl_class_imgs = list(self.class_dict[class_name]['unlbl'])

    if self._use_unlabel:
      lbl_class_imgs = unlbl_class_imgs
    else:
      lbl_class_imgs = list(self.class_dict[class_name]['lbl'])

    np.random.shuffle(lbl_class_imgs)
    np.random.shuffle(unlbl_class_imgs)

    k_train = self.al_instance.k_train
    k_test = self.al_instance.k_test
    k_unlbl = self.al_instance.k_unlbl
    msg = 'Num Image {} Support {} Query {}'.format(
        len(lbl_class_imgs), k_train, k_test)
    assert len(lbl_class_imgs) >= k_train + k_test, msg
    train = self._read_set(lbl_class_imgs[:k_train])
    test = self._read_set(lbl_class_imgs[k_train:k_train + k_test])

    # if unlabeled partition is not empty, get unlabeled images from there
    # otherwise, get from labeled partition
    if len(unlbl_class_imgs) > 0:
      assert len(unlbl_class_imgs) >= k_unlbl
      unlbl = self._read_set(unlbl_class_imgs[:k_unlbl])
    else:
      assert len(lbl_class_imgs) >= k_train + k_test + k_unlbl
      unlbl = self._read_set(
          lbl_class_imgs[k_train + k_test:k_train + k_test + k_unlbl])
    return train, test, unlbl

  def normalize(self, x):
    # return (x - 0.5) * 2.0
    return (x - self.mean_pix) / self.std_pix

  def _read_set(self, image_list):
    '''
      return matrix for all images in image_list
      '''
    data = []
    for image_file in image_list:
      x = self._read_from_cache(image_file)
      # x = self.normalize(x)
      data.append(x)

    if len(data) == 0:
      return np.zeros([0, 84, 84, 3])
    else:
      return np.stack(data)

  def _read_from_cache(self, idx):
    return self.img_data[idx] / 255.0

  def _concat_or_identity(self, big_set, small_set):
    if big_set is None:
      return small_set
    else:
      return DatasetT(
          data=np.concatenate((big_set.data, small_set.data)),
          labels=np.concatenate((big_set.labels, small_set.labels)))

  def _check_shape(self, dataset, n_class, n_items):
    s = dataset.data.shape
    assert s == (n_class * n_items, self.n_input, self.n_input, 3)
    assert s[0] == n_class * n_items

    return True

  def reset(self):
    pass

  @property
  def num_classes(self):
    return self._num_classes

  def get_size(self):
    """Gets the size of the supervised portion."""
    return self._num_lbl

  def get_size_test(self):
    """Gets the size of the unsupervised portion."""
    return self._num_unlbl

  def get_batch_idx(self, idx, forbid=None):
    """Gets a fully supervised training batch for classification.

    Returns: A tuple of
      x: Input image batch [N, H, W, C].
      y: Label class integer ID [N].
    """
    if forbid is None:
      x = self._read_from_cache(self._lbl_idx[idx])
      y = np.array([self._cls_label[kk] for kk in self._lbl_idx[idx]],
                   dtype=np.int64)
    else:
      classes = list(range(len(self.class_dict.keys())))
      for kk in forbid:
        classes.remove(kk)
      sel_classes = np.random.choice(classes, size=len(idx), replace=True)
      sel_classes, sel_classes_num = np.unique(sel_classes, return_counts=True)

      idx_new = []
      y = []
      for kk, knum in zip(sel_classes, sel_classes_num):
        class_name = self.idic[kk]
        img_list = self.class_dict[class_name]['lbl']
        img_ids = np.random.choice(img_list, size=knum, replace=False)
        idx_new.extend(img_ids)
        y.extend([kk] * knum)
      x = self._read_from_cache(idx_new)
      y = np.array(y)

    if self._split == 'train_phase_train':
      for jj in range(x.shape[0]):
        x[jj] = self._sess.run(
            self._rnd_process, feed_dict={self._rnd_process_plh: x[jj]})
    x = self.normalize(x)
    return x, y

  def get_batch_idx_test(self, idx):
    """Gets the test set (unlabeled set) for the fully supervised training."""
    x = self._read_from_cache(self._unlbl_idx[idx])
    y = np.array([self._cls_label[kk] for kk in self._unlbl_idx[idx]],
                 dtype=np.int64)
    # x = (x - 0.5) * 2.0
    x = self.normalize(x)
    return x, y

  def tf_preprocess(random_crop=True,
                    random_flip=True,
                    random_color=True,
                    whiten=False):
    image_size = 84
    inp = tf.placeholder(tf.float32, [image_size, image_size, 3])
    image = inp
    # image = tf.cast(inp, tf.float32)
    if random_crop:
      log.info("Apply random cropping")
      image = tf.image.resize_image_with_crop_or_pad(inp, image_size + 8,
                                                     image_size + 8)
      image = tf.random_crop(image, [image_size, image_size, 3])
    if random_flip:
      log.info("Apply random flipping")
      image = tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    if random_color:
      image = tf.image.random_brightness(image, max_delta=63. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    if whiten:
      log.info("Apply whitening")
      image = tf.image.per_image_whitening(image)
    return inp, image
