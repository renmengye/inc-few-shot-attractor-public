import cv2
import numpy as np
import six
import sys
import pickle as pkl

from tqdm import tqdm


def compress(path, output):
  with np.load(path, mmap_mode="r", encoding='latin1') as data:
    images = data["images"]
    array = []
    for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
      im = images[ii]
      im_str = cv2.imencode('.png', im)[1]
      array.append(im_str)
  with open(output, 'wb') as f:
    pkl.dump(array, f, protocol=pkl.HIGHEST_PROTOCOL)


def decompress(path, output):
  try:
    with open(output, 'rb') as f:
      array = pkl.load(f, encoding='bytes')
  except:
    with open(output, 'rb') as f:
      array = pkl.load(f)
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)


def main():
  if sys.argv[1] == 'compress':
    compress(sys.argv[2], sys.argv[3])
  elif sys.argv[1] == 'decompress':
    decompress(sys.argv[2], sys.argv[3])


if __name__ == '__main__':
  main()
