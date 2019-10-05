from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Episode(object):

  def __init__(self,
               x_train,
               y_train,
               x_test,
               y_test,
               x_unlabel=None,
               y_unlabel=None,
               y_train_str=None,
               y_test_str=None,
               y_sel=None):
    """Creates a miniImageNet episode.
    Args:
      x_train:  [N, ...]. Training data.
      y_train: [N]. Training label.
      x_test: [N, ...]. Testing data.
      y_test: [N]. Testing label.
    """
    self._x_train = x_train
    self._y_train = y_train
    self._x_test = x_test
    self._y_test = y_test
    self._x_unlabel = x_unlabel
    self._y_unlabel = y_unlabel
    self._y_train_str = y_train_str
    self._y_test_str = y_test_str
    self._y_sel = y_sel
    # self._rnd = np.random.RandomState(1234)

  # def next_batch(self, batch_size):
  #   assert x_train.shape[0] > batch_size
  #   _rnd.shuffle(np.arange(x_train.shape[0]))
  #   return self

  def next_batch(self):
    return self

  @property
  def x_train(self):
    return self._x_train

  @property
  def x_test(self):
    return self._x_test

  @property
  def y_train(self):
    return self._y_train

  @property
  def y_test(self):
    return self._y_test

  @property
  def x_unlabel(self):
    return self._x_unlabel

  @property
  def y_unlabel(self):
    return self._y_unlabel

  @property
  def y_train_str(self):
    return self._y_train_str

  @property
  def y_test_str(self):
    return self._y_test_str

  @property
  def y_sel(self):
    return self._y_sel
