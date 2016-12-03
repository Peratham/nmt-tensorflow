from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os

from translate.model import Model
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.client import device_lib
from time import time

tf.app.flags.DEFINE_string('output_dir', '/tmp', 'Output directory.')
tf.app.flags.DEFINE_string('train_dir', '/tmp', 'Training directory.')

_FLAGS = tf.app.flags.FLAGS
_DATA = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
         [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
_NUM_ITER = 20
_GPU = map(lambda x: x.name, filter(lambda d: d.device_type == 'GPU', 
                                    device_lib.list_local_devices()))
_CONFIG = tf.ConfigProto(allow_soft_placement=True,
                         log_device_placement=True)

def linebreak():
    return '-' * 50 + '\n'

def single_computation():
    with open(os.path.join(_FLAGS.output_dir, 'single_computation.out'), 'w') as f:
        graph = tf.Graph()
        t = time()
        with graph.as_default(), tf.device(_GPU[0]):
            model = Model(source_vocab_size=10, 
                          target_vocab_size=10,
                          buckets=[(3, 3), (6, 6)], 
                          size=32,
                          num_layers=2,
                          learning_rate=None,
                          max_gradient_norm=5.0, 
                          batch_size=32,
                          use_lstm=True, 
                          use_local=True,
                          optim='adam',
                          num_samples=None)
        f.write('Initializing Graph took %.3f s\n' %(time() - t))
        f.write(linebreak())
        
        with tf.Session(graph=graph, config=_CONFIG) as sess:
            t = time()
            sess.run(tf.initialize_all_variables())
            f.write('Initializing Variables took %.3f s\n' %(time() - t))
            
            t = time()
            for _ in xrange(_NUM_ITER):
                bucket_id = random.choice([0, 1])
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(_DATA, bucket_id)
                loss, _ = model.step(sess, 
                                     encoder_inputs, 
                                     decoder_inputs, 
                                     target_weights, 
                                     bucket_id, 
                                     False)
                f.write('Perplexity\t: %f\n' %(np.exp(loss)))
            
            f.write(linebreak())
            f.write('Average training time: %.3f s/iter\n' %((time() - t)/_NUM_ITER))

def twin_computation():
    with open(os.path.join(_FLAGS.output_dir, 'twin_computation.out'), 'w') as f:
        graph = tf.Graph()
        t = time()
        with graph.as_default():
            with tf.device(_GPU[0]), vs.variable_scope('graph_one') as gscope_1:
                model_one = Model(source_vocab_size=10, 
                                  target_vocab_size=10,
                                  buckets=[(3, 3), (6, 6)], 
                                  size=32,
                                  num_layers=2,
                                  learning_rate=None,
                                  max_gradient_norm=5.0, 
                                  batch_size=32,
                                  use_lstm=True, 
                                  use_local=True,
                                  optim='adam',
                                  scope=gscope_1.name,
                                  num_samples=None)
            with tf.device(_GPU[1]), vs.variable_scope('graph_two') as gscope_2:
                model_two = Model(source_vocab_size=10, 
                                  target_vocab_size=10,
                                  buckets=[(3, 3), (6, 6)], 
                                  size=32,
                                  num_layers=2,
                                  learning_rate=None,
                                  max_gradient_norm=5.0, 
                                  batch_size=32,
                                  use_lstm=True, 
                                  use_local=True,
                                  optim='adam',
                                  scope=gscope_2.name,
                                  num_samples=None)
        f.write('Initializing Graphs took %.3f s\n' %(time() - t))
        f.write(linebreak())
        
        with tf.Session(graph=graph, config=_CONFIG) as sess:
            t = time()
            sess.run(tf.initialize_all_variables())
            f.write('Initializing Variables took %.3f s\n' %(time() - t))
            
            t = time()
            for _ in xrange(_NUM_ITER):
                bucket_id = random.choice([0, 1])
                encoder_inputs, decoder_inputs, target_weights = model_one.get_batch(_DATA, bucket_id)
                
                of_one, if_one = model_one.step(sess, 
                                                encoder_inputs, 
                                                decoder_inputs, 
                                                target_weights, 
                                                bucket_id, 
                                                forward_only=False, 
                                                delayed=True)
                of_two, if_two = model_two.step(sess, 
                                                encoder_inputs, 
                                                decoder_inputs, 
                                                target_weights, 
                                                bucket_id, 
                                                forward_only=False, 
                                                delayed=True)
                output_feed = of_one + of_two
                
                input_feed = {}
                input_feed.update(if_one)
                input_feed.update(if_two)
                
                loss_one, _, loss_two, _ = sess.run(output_feed, input_feed)
                f.write('Perplexity: %f | %f\n' %(np.exp(loss_one), np.exp(loss_two)))
            
            f.write(linebreak())
            f.write('Average training time: %.3f s/iter\n' %((time() - t)/_NUM_ITER))

def main(_):
    single_computation()
    twin_computation()
    
if __name__ == '__main__':
    tf.app.run()
