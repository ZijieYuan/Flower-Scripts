# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

from PIL import Image
import os

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(dataset_tf_dir,if_training=True):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    dataset_tf_dir: the path that saves "validation_filenames.txt".

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  """
  filenames = []
  if if_training == True:
      with open(dataset_tf_dir+'/training_filenames.txt') as f:
          filenames = [line.rstrip('\n') for line in f]
  else:
      with open(dataset_tf_dir+'/validation_filenames.txt') as f:
          filenames = [line.rstrip('\n') for line in f]

  num_images = len(filenames)

  img = Image.open(filenames[0])
  img = img.resize((300, 300), Image.NEAREST)
  np_images = numpy.asarray(img, dtype='uint8')
  np_images = list(np_images)

  for i in range(1,num_images):
      print('Start reading image ',i)
      image = Image.open(filenames[i])
      image = image.resize((300, 300), Image.NEAREST)
      np_image = numpy.asarray(image, dtype='uint8')
      np_image = list(np_image)
      np_images = numpy.concatenate((np_images,np_image),axis=0)

  np_images = numpy.reshape(np_images,(-1,300,300,3))
  np_images = numpy.array(np_images)
  return np_images


def extract_labels(dataset_tf_dir, if_training=True):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    dataset_tf_dir: the path that saves "validation_filenames.txt".

  Returns:
    labels: a 1D uint8 numpy array.

  """
  filenames = []
  if if_training == True:
      with open(dataset_tf_dir+'/training_filenames.txt' , 'r') as f:
          filenames = [line.rstrip('\n') for line in f]
  else:
      with open(dataset_tf_dir+'/validation_filenames.txt') as f:
          filenames = [line.rstrip('\n') for line in f]

  class_names_to_ids = {}
  with open(dataset_tf_dir+'/class_names_to_ids.txt') as f2:
      for kv in [d.strip().split(':') for d in f2]:
          class_names_to_ids[kv[0]] = kv[1] 

  np_labels = []
  for im in filenames:
      path,folder_name = os.path.split(im)
      path,folder_name = os.path.split(path)
      label = int(class_names_to_ids[folder_name])
      np_labels.append(label)
  np_labels = numpy.array(np_labels)
  return np_labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    `dtype` can be either `uint8` to leave the input as `[0, 255]`, 
    or `float32` to rescale into `[0, 1]`.  Seed arg provides 
    for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

#    assert numpy.shape(images)[0] == numpy.shape(labels)[0], (
#        'images.shape: %s labels.shape: %s' % (numpy.shape(images), labels.shape))
#    self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)

    if reshape:
      assert images.shape[3] == 3
      images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(dataset_tf_dir,
                   dtype=dtypes.float32,
                   reshape=True,
                   seed=None):


  total_images = extract_images(dataset_tf_dir,if_training=True)
  total_labels = extract_labels(dataset_tf_dir,if_training=True)
  total_idx = [i for i in range(len(total_labels))]
  test_images = total_images[::10]
  test_labels = total_labels[::10]
  test_idx = total_idx[::10]
  train_idx = list(set(total_idx) - set(test_idx))
  train_images = total_images[numpy.array(train_idx)]
  train_labels = total_labels[numpy.array(train_idx)] 
  validation_images = extract_images(dataset_tf_dir,if_training=False)
  validation_labels = extract_labels(dataset_tf_dir,if_training=False)


  train = DataSet(
      train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
  validation = DataSet(
      validation_images,
      validation_labels,
      dtype=dtype,
      reshape=reshape,
      seed=seed)
  test = DataSet(
      test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

  return base.Datasets(train=train, validation=validation, test=test)


def load_flowers(train_dir='flowers-data'):
  return read_data_sets(dataset_tf_dir)
