from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.episode import Episode
from fewshot.utils import logger

log = logger.get()


class MetaDataset(object):

  def next(self):
    """Get a new episode training."""
    pass

  def stop(self):
    """Stops iterating."""
    pass


class RefinementMetaDataset(MetaDataset):
  """A few-shot learning dataset with refinement (unlabeled) training. images.
  """

  def __init__(self,
               split,
               nway,
               nshot,
               num_unlabel,
               num_distractor,
               num_test,
               label_ratio,
               shuffle_episode,
               seed,
               use_unlabel=False,
               image_size=84,
               crop_size=92):
    """Creates a meta dataset.
    Args:
      folder: String. Path to the dataset.
      split: String.
      nway: Int. N way classification problem, default 5.
      nshot: Int. N-shot classification problem, default 1.
      num_unlabel: Int. Number of unlabeled examples per class, default 2.
      num_distractor: Int. Number of distractor classes, default 0.
      num_test: Int. Number of query images, default 10.
      split_def: String. "vinyals" | "lake", using different split definitions.
      aug_90: Bool. Whether to augment training data by rotating 90 degrees.
      seed: Int. Random seed.
      use_unlabel: Bool. Use 'unlabeled' set for fewshot episode generation.
    """
    self._split = split
    self._nway = nway
    self._nshot = nshot
    self._num_unlabel = num_unlabel
    self._rnd = np.random.RandomState(seed)
    self._seed = seed
    self._num_distractor = 0
    log.warning("Number of distractors in each episode: {}".format(
        self._num_distractor))
    self._num_test = num_test
    self._label_ratio = label_ratio
    log.info('Label ratio {}'.format(self._label_ratio))
    self._shuffle_episode = shuffle_episode
    self._use_unlabel = use_unlabel

    if use_unlabel:
      assert self._num_distractor == 0
      assert self._num_unlabel == 0

    self.read_dataset()

    # Build a set for quick query.
    self._label_split_idx = np.array(self._label_split_idx)
    self._label_split_idx_set = set(list(self._label_split_idx))
    self._unlabel_split_idx = list(
        filter(lambda _idx: _idx not in self._label_split_idx_set,
               range(self._labels.shape[0])))
    self._unlabel_split_idx = np.array(self._unlabel_split_idx)
    if len(self._unlabel_split_idx) > 0:
      self._unlabel_split_idx_set = set(self._unlabel_split_idx)
    else:
      self._unlabel_split_idx_set = set()

    num_label_cls = len(self._label_str)
    self._num_classes = num_label_cls
    num_ex = self._labels.shape[0]
    ex_ids = np.arange(num_ex)
    self._label_idict = {}
    for cc in range(num_label_cls):
      self._label_idict[cc] = ex_ids[self._labels == cc]
    self._nshot = nshot
    with tf.device('/cpu:0'):
      self._crop_process_plh, self._crop_process = self.tf_preprocess(
          image_size=image_size,
          crop_size=crop_size,
          random_crop=False,
          random_flip=False,
          random_color=False)
      if self._split.startswith('train'):
        self._rnd_process_plh, self._rnd_process = self.tf_preprocess(
            image_size=image_size, crop_size=crop_size, random_color=False)
      else:
        self._rnd_process_plh, self._rnd_process = self.tf_preprocess(
            image_size=image_size,
            crop_size=crop_size,
            random_crop=False,
            random_flip=False,
            random_color=False)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=sess_config)

  def read_dataset(self):
    """Reads data from folder or cache."""
    raise NotImplemented()

  def label_split(self):
    """Gets label/unlabel image splits.
    Returns:
      labeled_split: List of int.
    """
    log.info('Label split using seed {:d}'.format(self._seed))
    rnd = np.random.RandomState(self._seed)
    num_label_cls = len(self._label_str)
    num_ex = self._labels.shape[0]
    ex_ids = np.arange(num_ex)

    labeled_split = []
    for cc in range(num_label_cls):
      cids = ex_ids[self._labels == cc]
      rnd.shuffle(cids)
      labeled_split.extend(cids[:int(len(cids) * self._label_ratio)])
    log.info("Total number of classes {}".format(num_label_cls))
    log.info("Labeled split {}".format(len(labeled_split)))
    log.info("Total image {}".format(num_ex))
    return sorted(labeled_split)

  def next_increment(self):
    pass

  def next(self,
           within_category=False,
           catcode=None,
           class_b=None,
           random_shots=False):
    """Gets a new episode.
    within_category: bool. Whether or not to choose the N classes
    to all belong to the same more general category.
    (Only applicable for datasets with self._category_labels defined).

    within_category: bool. Whether or not to restrict the episode's classes
    to belong to the same general category (only applicable for JakeImageNet).
    If True, a random general category will be chosen, unless catcode is set.

    catcode: str. (e.g. 'n02795169') if catcode is provided (is not None),
    then the classes chosen for this episode will be restricted
    to be synsets belonging to the more general category with code catcode.
    """

    if class_b is None:
      if within_category or catcode is not None:
        assert hasattr(self, "_category_labels")
        assert hasattr(self, "_category_label_str")
        if catcode is None:
          # Choose a category for this episode's classes
          cat_idx = np.random.randint(len(self._category_label_str))
          catcode = self._catcode_to_syncode.keys()[cat_idx]
        cat_synsets = self._catcode_to_syncode[catcode]
        cat_synsets_str = [self._syncode_to_str[code] for code in cat_synsets]
        allowable_inds = []
        for str in cat_synsets_str:
          allowable_inds.append(np.where(np.array(self._label_str) == str)[0])
        class_seq = np.array(allowable_inds).reshape((-1))
      else:
        num_label_cls = len(self._label_str)
        class_seq = np.arange(num_label_cls)
      self._rnd.shuffle(class_seq)
    else:
      num_label_cls = len(self._label_str)
      class_seq = np.arange(num_label_cls)
      class_seq = class_b

    train_img_ids = []
    train_labels = []
    test_img_ids = []
    test_labels = []

    train_unlabel_img_ids = []
    non_distractor = []

    train_labels_str = []
    test_labels_str = []

    is_training = self._split.startswith("train")
    assert is_training or self._split in ["val", "test"]

    for ii in range(self._nway + self._num_distractor):

      cc = class_seq[ii]
      # print(cc, ii < self._nway)
      _ids = self._label_idict[cc]

      # Split the image IDs into labeled and unlabeled.
      _label_ids = list(
          filter(lambda _id: _id in self._label_split_idx_set, _ids))
      _unlabel_ids = list(
          filter(lambda _id: _id not in self._label_split_idx_set, _ids))
      self._rnd.shuffle(_label_ids)
      self._rnd.shuffle(_unlabel_ids)

      # Add support set and query set (not for distractors).

      if self._use_unlabel:
        _img_ids = _unlabel_ids
      else:
        _img_ids = _label_ids

      if ii < self._nway:
        if random_shots:
          nshot = np.random.choice([1, 2, 3, 5, 10])
        else:
          nshot = self._nshot
        train_img_ids.extend(_img_ids[:nshot])

        # Use the rest of the labeled image as queries, if num_test = -1.
        QUERY_SIZE_LARGE_ERR_MSG = (
            "Query + reference should be less than labeled examples." +
            "Num labeled {} Num test {} Num shot {}".format(
                len(_img_ids), self._num_test, nshot))
        assert nshot + self._num_test <= len(
            _img_ids), QUERY_SIZE_LARGE_ERR_MSG

        if self._num_test == -1:
          if is_training:
            num_test = len(_img_ids) - nshot
          else:
            num_test = len(_img_ids) - nshot - self._num_unlabel
        else:
          num_test = self._num_test
          if is_training:
            assert num_test <= len(_img_ids) - nshot
          else:
            assert num_test <= len(_img_ids) - self._num_unlabel - nshot

        test_img_ids.extend(_img_ids[nshot:nshot + num_test])
        train_labels.extend([ii] * nshot)
        train_labels_str.extend([self._label_str[cc]] * nshot)
        test_labels.extend([ii] * num_test)
        test_labels_str.extend([self._label_str[cc]] * num_test)
        non_distractor.extend([1] * self._num_unlabel)
      else:
        non_distractor.extend([0] * self._num_unlabel)

      # Add unlabeled images here.
      if is_training:
        # Use labeled, unlabeled split here for refinement.
        train_unlabel_img_ids.extend(_unlabel_ids[:self._num_unlabel])

      else:
        # Copy test set for refinement.
        # This will only work if the test procedure is rolled out in
        # a sequence.
        train_unlabel_img_ids.extend(
            _img_ids[nshot + num_test:nshot + num_test + self._num_unlabel])

    train_img = self.get_images(train_img_ids)
    train_unlabel_img = self.get_images(train_unlabel_img_ids)
    test_img = self.get_images(test_img_ids)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_labels_str = np.array(train_labels_str)
    test_labels_str = np.array(test_labels_str)
    non_distractor = np.array(non_distractor)

    test_ids_set = set(test_img_ids)
    for _id in train_unlabel_img_ids:
      assert _id not in test_ids_set

    if self._shuffle_episode:
      # Shuffle the sequence order in an episode. Very important for RNN based
      # meta learners.
      train_idx = np.arange(train_img.shape[0])
      self._rnd.shuffle(train_idx)
      train_img = train_img[train_idx]
      train_labels = train_labels[train_idx]

      train_unlabel_idx = np.arange(train_unlabel_img.shape[0])
      self._rnd.shuffle(train_unlabel_idx)
      train_unlabel_img = train_unlabel_img[train_unlabel_idx]

      test_idx = np.arange(test_img.shape[0])
      self._rnd.shuffle(test_idx)
      test_img = test_img[test_idx]
      test_labels = test_labels[test_idx]

    if 'train' in self._split and self._split != 'train_b':
      y_sel = class_seq[:self._nway]
    else:
      y_sel = None

    # print('train img before crop', train_img.max(), train_img.min())
    train_img2 = []
    test_img2 = []
    for _img in train_img:
      _img2 = self._sess.run(
          self._crop_process, feed_dict={self._crop_process_plh: _img})
      train_img2.append(_img2)
    for _img in test_img:
      _img2 = self._sess.run(
          self._crop_process, feed_dict={self._crop_process_plh: _img})
      test_img2.append(_img2)
    train_img = np.stack(train_img2, axis=0)
    test_img = np.stack(test_img2, axis=0)

    if type(train_unlabel_img) == list:
      if len(train_unlabel_img) > 0:
        train_unlabel_img = np.stack(train_unlabel_img, axis=0)
      else:
        train_unlabel_img = np.array(train_unlabel_img)
    train_img = self.normalize(train_img)
    test_img = self.normalize(test_img)
    train_unlabel_img = self.normalize(train_unlabel_img)

    return Episode(
        train_img,
        train_labels,
        test_img,
        test_labels,
        x_unlabel=train_unlabel_img,
        y_unlabel=non_distractor,
        y_train_str=train_labels_str,
        y_test_str=test_labels_str,
        y_sel=y_sel)

  def reset(self):
    self._rnd = np.random.RandomState(self._seed)

  def get_size(self):
    """Gets the size of the supervised portion."""
    return len(self._label_split_idx)

  def get_size_test(self):
    """Gets the size of the unsupervised portion."""
    return len(self._unlabel_split_idx)

  def normalize(self, x):
    return (x - 0.5) * 2.0
    # return (x - self.mean_pix) / self.std_pix

  def get_batch_idx(self, idx, forbid=None):
    """Gets a fully supervised training batch for classification.

    Returns: A tuple of
      x: Input image batch [N, H, W, C].
      y: Label class integer ID [N].
    """
    if forbid is None:
      x = self.get_images(self._label_split_idx[idx])
      y = self._labels[self._label_split_idx[idx]]
    else:
      # log.info('Forbid {}'.format(forbid))
      classes = list(range(len(self._label_idict.keys())))
      for kk in forbid:
        classes.remove(kk)
      sel_classes = np.random.choice(classes, size=len(idx), replace=True)
      sel_classes, sel_classes_num = np.unique(sel_classes, return_counts=True)

      idx_new = []
      y = []
      for kk, knum in zip(sel_classes, sel_classes_num):
        _ids = self._label_idict[kk]
        img_ids = list(
            filter(lambda _id: _id in self._label_split_idx_set, _ids))
        img_ids = np.random.choice(img_ids, size=knum, replace=False)
        idx_new.extend(img_ids)
        y.extend([kk] * knum)
      x = self.get_images(idx_new)
      y = np.array(y)

    x2 = []
    for x_ in x:
      x2_ = self._sess.run(
          self._rnd_process, feed_dict={self._rnd_process_plh: x_})
      x2.append(x2_)
    x2 = np.stack(x2, axis=0)
    x2 = self.normalize(x2)
    return x2, y

  def get_batch_idx_test(self, idx, forbid=None):
    """Gets the test set (unlabeled set) for the fully supervised training."""
    if forbid is None:
      x = self.get_images(self._unlabel_split_idx[idx])
      y = self._labels[self._unlabel_split_idx[idx]]
    else:
      # log.info('Forbid {}'.format(forbid))
      classes = list(range(len(self._label_idict.keys())))
      for kk in forbid:
        classes.remove(kk)
      sel_classes = np.random.choice(classes, size=len(idx), replace=True)
      sel_classes, sel_classes_num = np.unique(sel_classes, return_counts=True)

      idx_new = []
      y = []
      for kk, knum in zip(sel_classes, sel_classes_num):
        _ids = self._label_idict[kk]
        img_ids = list(
            filter(lambda _id: _id not in self._label_split_idx_set, _ids))
        img_ids = np.random.choice(img_ids, size=knum, replace=False)
        idx_new.extend(img_ids)
        y.extend([kk] * knum)
      x = self.get_images(idx_new)
      y = np.array(y)

    x2 = []
    for x_ in x:
      x2_ = self._sess.run(
          self._rnd_process, feed_dict={self._rnd_process_plh: x_})
      x2.append(x2_)
    x2 = np.stack(x2, axis=0)
    x2 = self.normalize(x2)
    return x, y

  def tf_preprocess(self,
                    image_size=84,
                    crop_size=92,
                    random_crop=True,
                    random_flip=True,
                    random_color=True,
                    whiten=False):
    inp = tf.placeholder(tf.uint8, [None, None, 3])
    image = tf.realdiv(tf.cast(inp, tf.float32), 255.0)
    # image = debug_identity(image)
    if random_crop:
      log.info("Apply random cropping")
      image = tf.image.resize_image_with_crop_or_pad(image, crop_size,
                                                     crop_size)
      image = tf.random_crop(image, [image_size, image_size, 3])
    else:
      image = tf.image.resize_image_with_crop_or_pad(image, image_size,
                                                     image_size)
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

  @property
  def num_classes(self):
    return self._num_classes
