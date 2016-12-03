from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os

from translate import data_utils

from tensorflow.python.client import device_lib
from tensorflow.python.ops import variable_scope as vs

from translate.attn_lib import embedding_attention_s2s as s2s
from translate.model_utils import full_sequence_loss as fsl
from translate.model_utils import sampled_sequence_loss as ssl
from translate.model_utils import model_with_buckets

from time import time

tf.app.flags.DEFINE_string('output_dir', '/tmp', 'Output directory.')
tf.app.flags.DEFINE_integer('num_gpus', 2, 'Number of GPUs')

_FLAGS = tf.app.flags.FLAGS
_BUCKETS = [(3, 3), (6, 6)]
_DATA = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
         [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
_NUM_ITER = 10
_GPU = map(lambda x: x.name, filter(lambda d: d.device_type == 'GPU',
                                    device_lib.list_local_devices()))
_CONFIG = tf.ConfigProto(allow_soft_placement=True,
                         log_device_placement=True)


def linebreak():
    return '-' * 50 + '\n'


def average_gradients(tower_grads, buckets):
    bucket_grads = []
    for i, _ in enumerate(buckets):
        average_grads = []
        for grad_and_vars in zip(tower_grads[0][i], tower_grads[1][i]):
            grad = tf.reduce_mean(tf.concat(0, [tf.expand_dims(g, 0) for g, _ in grad_and_vars]), 0)
            average_grads.append((grad, grad_and_vars[0][1]))
        bucket_grads.append(average_grads)
    return bucket_grads

def gradients(tower_losses, buckets, opt, global_step, max_gradient_norm=5.0):
    params = tf.trainable_variables()
    g0_losses, g1_losses = tower_losses
    updates = []
    for b in xrange(len(buckets)):
        grad_0 = tf.gradients(g0_losses[b], params)
        grad_1 = tf.gradients(g1_losses[b], params)
        grads = [tf.reduce_mean(tf.concat(0, [tf.expand_dims(g0, 0), tf.expand_dims(g1, 0)]), 0)
                 for g0, g1 in zip(grad_0, grad_1)]
        clipped_grads, _ = tf.clip_by_global_norm(grads, max_gradient_norm)
        updates.append(opt.apply_gradients(zip(clipped_grads, params), global_step=global_step))
    return updates

def _get_learning_rate(optimizer='adam'):
    return {
        'sgd': 0.5,
        'adam': 0.01,
        'rmsprop': 0.1,
        'adagrad': 0.1
    }[optimizer]


def _get_optimizer(learning_rate, optimizer='sgd'):
    """
    Internal method to retrieve optimizer.
    """
    opts = {
        'sgd': tf.train.GradientDescentOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adam': lambda x: tf.train.AdamOptimizer(learning_rate=x, epsilon=1e-10),
        'rmsprop': lambda x: tf.train.RMSPropOptimizer(learning_rate=x, momentum=0.5)
    }

    if optimizer not in opts:
        raise KeyError('Invalid optimizer. Must be one of sgd, rmsprop or adam.')

    return opts[optimizer](learning_rate)


def train(source_vocab_size,
          target_vocab_size,
          buckets,
          size,
          num_layers=3,
          optim='adam',
          learning_rate=None,
          learning_rate_decay_factor=0.99,
          batch_size=512,
          max_gradient_norm=5.0,
          num_samples=None,
          use_lstm=True,
          use_lstm_peepholes=False,
          use_local=True,
          dtype=tf.float32):
    f = open(os.path.join(_FLAGS.output_dir, 'multi_gpu.out'), 'w')
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        learning_rate = _get_learning_rate(optim) if learning_rate is None else learning_rate
        lr = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        learning_rate_decay_op = lr.assign(lr * learning_rate_decay_factor)

        opt = _get_optimizer(lr, optimizer=optim)

        #tower_grads = []
        tower_inputs = []
        tower_losses = []

        t = time()
        for i in xrange(_FLAGS.num_gpus):
            with tf.device(_GPU[i]), tf.name_scope('tower_%d' % i) as global_scope:
                with vs.variable_scope('inputs_%d' % i, reuse=False) as input_scope:
                    enc_inputs, dec_inputs, target_weights, targets = scoped_inputs(buckets)
                    tower_inputs.append((enc_inputs, dec_inputs, target_weights))
                
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                tower_loss = inference(enc_inputs,
                                       dec_inputs,
                                       target_weights,
                                       targets,
                                       source_vocab_size,
                                       target_vocab_size,
                                       size,
                                       buckets,
                                       num_layers=num_layers,
                                       num_samples=num_samples,
                                       use_lstm=use_lstm,
                                       use_lstm_peepholes=use_lstm_peepholes,
                                       use_local=use_local)
                
                tower_losses.append(tower_loss)
                #tower_grads.append([opt.compute_gradients(loss) for loss in tower_loss])

        f.write('Initializing Graph took %.3fs\n' % (time() - t))
        f.write(linebreak())
        
        updates = gradients(tower_losses, buckets, opt, global_step, max_gradient_norm)

        #grads = average_gradients(tower_grads, buckets)
        #updates = [opt.apply_gradients(grad, global_step=global_step) for grad in grads]

        with tf.Session(graph=graph, config=_CONFIG) as sess:
            sess.run(tf.global_variables_initializer())

            t = time()
            for step in xrange(_NUM_ITER):
                k = random.choice([0, 1])

                enc_in_0, dec_in_0, tw_0 = get_batch(_DATA, buckets, k, batch_size)
                enc_in_1, dec_in_1, tw_1 = get_batch(_DATA, buckets, k, batch_size)

                encoder_size, decoder_size = buckets[k]
                input_feed = {}

                enc_inputs, dec_inputs, target_weights = tower_inputs[0]

                input_feed.update({enc_inputs[i].name: enc_in_0[i] for i in xrange(encoder_size)})
                input_feed.update({dec_inputs[i].name: dec_in_0[i] for i in xrange(decoder_size)})
                input_feed.update({target_weights[i].name: tw_0[i] for i in xrange(decoder_size)})
                last_target = dec_inputs[decoder_size].name
                input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)

                enc_inputs, dec_inputs, target_weights = tower_inputs[1]

                input_feed.update({enc_inputs[i].name: enc_in_1[i] for i in xrange(encoder_size)})
                input_feed.update({dec_inputs[i].name: dec_in_1[i] for i in xrange(decoder_size)})
                input_feed.update({target_weights[i].name: target_weights[i] for i in xrange(decoder_size)})
                last_target = dec_inputs[decoder_size].name
                input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)

                loss_1, loss_2, _ = sess.run([tower_losses[0][k] tower_losses[1][k], updates[k]], input_feed)
                f.write('Perplexity\t: %f\n' % (np.exp((loss_1+loss_2)/2)))

            f.write(linebreak())
            f.write('Average training time: %.3f s/iter\n' % ((time() - t) / _NUM_ITER))
        f.close()


def _get_loss(num_samples, target_vocab_size, proj_w_t, proj_w, proj_b):
    """
    Internal method to retrieve loss function.
    """
    if num_samples is not None and (0 < num_samples < target_vocab_size):
        # Sampled Softmax Loss.
        return lambda x, y, z: ssl(x, y, z, proj_w_t, proj_b, target_vocab_size, num_samples)
    else:
        # Full Softmax Loss.
        return lambda x, y, z: fsl(x, y, z, proj_w, proj_b)


def scoped_inputs(buckets, dtype=tf.float32):
    # Define naming function.
    name_f = lambda name, key: '{0}{1}'.format(name, key)

    # Define the input and weight placeholders.
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in xrange(buckets[-1][0]):
        encoder_inputs.append(tf.placeholder(tf.int32,
                                             shape=[None],
                                             name=name_f('encoder', i)))
    for i in xrange(buckets[-1][1] + 1):
        decoder_inputs.append(tf.placeholder(tf.int32,
                                             shape=[None],
                                             name=name_f('decoder', i)))
        target_weights.append(tf.placeholder(dtype,
                                             shape=[None],
                                             name=name_f('weight', i)))

    targets = decoder_inputs[1:]

    return encoder_inputs, decoder_inputs, target_weights, targets


def inference(enc_inputs,
              dec_inputs,
              target_weights,
              targets,
              source_vocab_size,
              target_vocab_size,
              size,
              buckets,
              num_layers=3,
              num_samples=None,
              use_lstm=True,
              use_lstm_peepholes=False,
              use_local=True):
    proj_w_t = tf.Variable(tf.random_normal([target_vocab_size, size]) / tf.sqrt(float(size)),
                           name='proj_w')
    proj_w = tf.transpose(proj_w_t)
    proj_b = tf.Variable(tf.random_normal([target_vocab_size]),
                         name='proj_b')
    output_proj = (proj_w, proj_b)

    # Creating the RNN cells.
    cell = None
    if use_lstm:
        cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=use_lstm_peepholes)
    else:
        cell = tf.nn.rnn_cell.GRUCell(size)

    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    # Define the Sequence-to-Sequence function with Input Embedding and Attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        return s2s(encoder_inputs,
                   decoder_inputs,
                   cell,
                   source_vocab_size,
                   target_vocab_size,
                   size,
                   output_projection=output_proj,
                   feed_previous=do_decode,
                   use_lstm=use_lstm,
                   local_p=use_local)

    # Training outputs and losses.
    loss_f = _get_loss(num_samples, target_vocab_size, proj_w_t, proj_w, proj_b)
    _, losses = model_with_buckets(enc_inputs,
                                   dec_inputs,
                                   lambda x, y: seq2seq_f(x, y, False),
                                   buckets,
                                   targets=targets,
                                   weights=target_weights,
                                   loss_f=loss_f)

    return losses


def get_batch(data, buckets, bucket_id, batch_size):
    """
    Get a batch from a specific bucket <bucket_id> in the dataset.
    """

    # Initializations.
    enc_size, dec_size = buckets[bucket_id]
    batch_enc = np.zeros((enc_size, batch_size), dtype=np.int32)
    batch_dec = np.zeros((dec_size, batch_size), dtype=np.int32)
    batch_weights = np.ones((dec_size, batch_size), dtype=np.float32)
    batch_dec[0, :] = data_utils._GO_ID

    # Gather Batch Data.
    for i in xrange(batch_size):
        enc_data, dec_data = random.choice(data[bucket_id])
        batch_enc[-len(enc_data):, i] = list(reversed(enc_data))
        batch_dec[1:len(dec_data) + 1, i] = dec_data

    # Set the Batch Weights.
    batch_weights[:-1, :] *= batch_dec[1:, :] > 0
    batch_weights[-1, :] = 0.0

    return batch_enc, batch_dec, batch_weights


def main(_):
    train(source_vocab_size=10,
          target_vocab_size=10,
          buckets=_BUCKETS,
          size=32,
          num_layers=2,
          batch_size=32,
          use_lstm=True,
          use_local=True,
          optim='adam',
          num_samples=None)


if __name__ == '__main__':
    tf.app.run()
