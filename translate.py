#################################################################################
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.                   #
# Modified by Pragaash Ponnusamy (2016) under the Apache License.               #
#                                                                               #
# Licensed under the Apache License, Version 2.0 (the "License");               #
# you may not use this file except in compliance with the License.              #
# You may obtain a copy of the License at                                       #
#                                                                               #
#     http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                               #
# Unless required by applicable law or agreed to in writing, software           #
# distributed under the License is distributed on an "AS IS" BASIS,             #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.      #
# See the License for the specific language governing permissions and           #
# limitations under the License.                                                #
#################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
import sys
import random
from bisect import bisect_right as br
from time import time
from time import localtime
from time import strftime

import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.python.ops import variable_scope as vs

from translate.model import Model

tf.app.flags.DEFINE_string('data_dir', '/tmp', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', '/tmp', 'Training directory.')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 200, 'Steps per checkpoint.')

# Global Constants.
_EN_DATA = 'giga-fren.r2.tok.ids50000.en'
_FR_DATA = 'giga-fren.r2.tok.ids50000.fr'
_EN_VOCAB = 50000
_FR_VOCAB = 50000
_FLAGS = tf.app.flags.FLAGS
_CONFIG = tf.ConfigProto(allow_soft_placement=True)
_GPU = map(lambda x: x.name, filter(lambda d: d.device_type == 'GPU', list_local_devices()))
_BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]
_MAX_EN = 40
_MAX_FR = 50


def get_int_seq(line):
    """
    Convert a string of space-separated values (SSV) into a list
    of integers.
    """
    return map(int, line.strip().split())


def read_into_buckets(en_ids_path, fr_ids_path, print_every=100000):
    """
    Generate dataset by reading data into buckets.
    """
    data_set = map(lambda x: [], _BUCKETS)
    counter = 0
    en_buckets, fr_buckets = zip(*_BUCKETS)
    with open(en_ids_path, 'r') as en:
        with open(fr_ids_path, 'r') as fr:
            en_line, fr_line = en.readline(), fr.readline()
            while en_line and fr_line:
                counter += 1
                if counter % print_every == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                en_sequence, fr_sequence = get_int_seq(en_line), get_int_seq(fr_line)
                en_seq_len, fr_seq_len = len(en_sequence), len(fr_sequence)
                if 0 < en_seq_len < _MAX_EN and 0 < fr_seq_len < _MAX_FR:
                    b_id = max(br(en_buckets, en_seq_len), br(fr_buckets, fr_seq_len))
                    data_set[b_id].append([en_sequence, fr_sequence])
                en_line, fr_line = en.readline(), fr.readline()
    return data_set


def get_data(binary_path=None, en_ids_path=None, fr_ids_path=None):
    """
    Load data from binary if exists, otherwise call subroutine to read tokenized
    data into buckets.
    """
    data_set = None
    if binary_path is not None and os.path.exists(binary_path):
        with open(binary_path, 'rb') as binary:
            data_set = pickle.load(binary)
    else:
        assert en_ids_path is not None and fr_ids_path is not None, 'Tokenized File Not Found!'
        data_set = read_into_buckets(en_ids_path, fr_ids_path)
    return data_set


def get_weights(data):
    std = np.sqrt(np.asarray(map(len, data)))
    return std.cumsum() / std.sum()


def linebreak():
    return '-' * 50 + '\n'

def train():
    # Graph Creation.
    graph = tf.Graph()
    t = time()
    with graph.as_default():
        model = Model(source_vocab_size=_EN_VOCAB,
                      target_vocab_size=_FR_VOCAB,
                      buckets=_BUCKETS,
                      size=512,
                      num_layers=3,
                      batch_size=256,
                      use_lstm=True,
                      use_local=True,
                      optim='adam',
                      num_samples=None)
    print(linebreak())
    print('Initializing Graphs took %.3f s\n' % (time() - t))
    print(linebreak())
    sys.stdout.flush()

    with tf.Session(graph=graph, config=_CONFIG) as sess:

        # Initializations.
        t = time()
        save_path = os.path.join(_FLAGS.train_dir, 'model')
        model.load(sess, save_path)
        print('Initializing Variables took %.3f s\n' % (time() - t))
        print(linebreak())
        sys.stdout.flush()
        
        # Gather Data.
        dataset = get_data(en_ids_path=os.path.join(_FLAGS.data_dir, _EN_DATA),
                           fr_ids_path=os.path.join(_FLAGS.data_dir, _FR_DATA))
        intervals = get_weights(dataset)

        # Book-keeping.
        step_time = 0.0
        batch_loss = 0.0
        current_step = 0
        prev_losses = []

        # Training.
        while True:
            bucket_id = np.abs(np.random.rand() - intervals).argmin()
            start_time = time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(dataset, bucket_id)
            loss, _ = model.step(sess,
                                 encoder_inputs,
                                 decoder_inputs,
                                 target_weights,
                                 bucket_id,
                                 forward_only=False)
            step_time += (time() - start_time) / _FLAGS.steps_per_checkpoint
            batch_loss += (loss/_FLAGS.steps_per_checkpoint)
            current_step += 1

            if current_step % _FLAGS.steps_per_checkpoint == 0:
                perplexity = np.exp(batch_loss)

                if len(prev_losses) > 2 and batch_loss > max(prev_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                prev_losses.append(batch_loss)
                
                wall_time = strftime('%m/%d %H:%M %p', localtime())
                print('> wall time: %s current-step: %d step-time: %.3f perplexity: %f' %(wall_time, current_step, step_time, perplexity))

                step_time = 0.0
                batch_loss = 0.0

                if current_step % (5 * _FLAGS.steps_per_checkpoint) == 0:
                    model.save(sess, save_path)
                sys.stdout.flush()

def self_test_model():
    """
    Test the translation model.
    """

    print("Self-test for neural translation model.")
    linebreak()

    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/cpu:0'):
            t = time()
            # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
            model = Model(source_vocab_size=10,
                          target_vocab_size=10,
                          buckets=[(3, 3), (6, 6)],
                          size=32,
                          num_layers=2,
                          learning_rate=None,
                          max_gradient_norm=5.0,
                          batch_size=32,
                          use_lstm=True,
                          optim='adam',
                          num_samples=None)

            print("Initializing Model took %.6fs" % (time() - t))
            linebreak()

    with tf.Session(graph=graph) as sess:
        t = time()
        sess.run(tf.initialize_all_variables())
        print("Initializing Variables took %.6fs" % (time() - t))
        linebreak()

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        num_iter = 20

        print('Using Learning Rate: %.2f' % (model.learning_rate.eval()))
        linebreak()

        t = time()
        # Train the fake model for 5 steps.
        for _ in xrange(num_iter):
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
            loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            print('Perplexity: %f' % (np.exp(loss)))
        linebreak()
        print("Average training time: %.6fs/iter" % ((time() - t) / num_iter))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
