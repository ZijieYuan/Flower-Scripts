# Copyright 2016 Norman Heckscher. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST dimensionality reduction with TensorFlow and TensorBoard.
This demonstrates the functionality of the TensorBoard Embedding Visualization dashboard using Flowers.
https://www.tensorflow.org/versions/r0.12/how_tos/embedding_viz/index.html#tensorboard-embedding-visualization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf
sys.path.append('/home/zijieyuan/miniconda3/envs/tensorflow/lib/tensorflow/examples/tutorials/flowers')
import input_data_zj
FLAGS = None

def generate_embeddings():
    # Import data
    print('Start reading data')
    flowers = input_data_zj.read_data_sets(FLAGS.data_dir)
    print('Finish reading data')
    sess = tf.InteractiveSession()

    # Input set for Embedded TensorBoard visualization
    # Performed with cpu to conserve memory and processing power
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.stack(flowers.test.images[:FLAGS.max_steps], axis=0), trainable=False, name='embedding')

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/projector', sess.graph)

    # Add embedding tensorboard visualization. Need tensorflow version
    # >= 0.12.0RC0
    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = os.path.join(FLAGS.log_dir + '/projector/labels.tsv')
    embed.sprite.image_path = os.path.join(FLAGS.log_dir + '/projector/master.jpg')

    # Specify the width and height of a single thumbnail.
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)

    saver.save(sess, os.path.join(
        FLAGS.log_dir, 'projector/a_model.ckpt'), global_step=FLAGS.max_steps)

##### The "metadata.tsv" ("labels.tsv") file can be generated along with "master.jpg" by "spriter.py"
##### It's better not to create this file here to ensure the label (labels.tsv) order is coherent with the sprite image (master.jpg)
"""
def generate_metadata_file():
    # Import data
    flowers = input_data_zj.read_data_sets(FLAGS.data_dir)
    def save_metadata(file):
        L = flowers.test.labels
        with open(file, 'w') as f:
            for i in range(len(L)):
                f.write('{}\n'.format(L[i]))

    save_metadata(FLAGS.log_dir + '/projector/metadata.tsv')
"""
#####

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir + '/projector'):
        tf.gfile.DeleteRecursively(FLAGS.log_dir + '/projector')
        tf.gfile.MkDir(FLAGS.log_dir + '/projector')
    tf.gfile.MakeDirs(FLAGS.log_dir  + '/projector') # fix the directory to be created
#    generate_metadata_file()
    print('Start "generate_embeddings()"')
    generate_embeddings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--data_dir', type=str, default='/home/zijieyuan/flowers',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/home/zijieyuan/flowers',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
