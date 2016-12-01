from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from time import time

from tensorflow.python.ops import variable_scope as vs


def basic_graph():
    batch = tf.placeholder(tf.float32, [256, 512], name='batch_data_matrix')
    with vs.variable_scope('full_con') as scope:
        activation = batch
        for i in range(100):
            with vs.variable_scope(scope, reuse=True if i > 0 else None):
                W_a = tf.get_variable('W_a', initializer=tf.random_normal([512, 512], stddev=0.1))
                b_a = tf.get_variable('b_a', initializer=tf.random_normal([512], stddev=0.1))
                activation = tf.sigmoid(tf.matmul(activation, W_a) + b_a)
    return batch, activation

def single_compute(data):
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        batch, activation = basic_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        return sess.run(activation, {batch: data})
    
def twin_compute(data):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/gpu:0'), vs.variable_scope('graph_1') as gscope_1:
            batch_1, activation_1 = basic_graph()
        with tf.device('/gpu:1'), vs.variable_scope('graph_2') as gscope_2:
            batch_2, activation_2 = basic_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        out_1, out_2 = sess.run([activation_1, activation_2],
                                {batch_1: data, batch_2: data})
    return {'g1_res': out_1, 'g2_res': out_2}

def sequential_compute(data):
    graph_1 = tf.Graph()
    with graph_1.as_default(), tf.device('/gpu:0'):
        batch_1, activation_1 = basic_graph()
    graph_2 = tf.Graph()
    with graph_2.as_default(), tf.device('/gpu:1'):
        batch_2, activation_2 = basic_graph()
    with tf.Session(graph=graph_1) as sess:
        sess.run(tf.initialize_all_variables())
        out_1 = sess.run(activation_1, {batch_1: data})
    with tf.Session(graph=graph_2) as sess:
        sess.run(tf.initialize_all_variables())
        out_2 = sess.run(activation_2, {batch_2: data})
    return {'g1_res': out_1, 'g2_res': out_2}

if __name__ == '__main__':
    
    data = np.random.randn(256, 512).astype(float32)
    
    print('- ' * 50)
    
    t = time()
    res = single_compute(data)
    print("Single Computation took %.6fs" %(time()-t))
    
    t = time()
    res = sequential_compute(data)
    print("Sequential Computation took %.6fs" %(time()-t))
    
    t = time()
    res = twin_compute(data)
    print("Twin Computation took %.6fs" %(time()-t))
    
    print('- ' * 50)