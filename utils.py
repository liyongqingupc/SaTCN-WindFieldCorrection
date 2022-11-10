import tensorflow as tf
import cv2
import numpy as np
import constants as c
save_dir = '../save'

def read_data(files, batch_size, histlen, futulen):
    m = 1  # the num of input channel
    data = np.empty([batch_size, histlen + 1 + futulen, c.data_height, c.data_width, m])
    for i in range(batch_size):
        data_single = np.empty([histlen + 1 + futulen, c.data_height, c.data_width, m])
        for j in range(histlen + 1 + futulen):
            #### 25-40N,110-125E ###
            #data_single[j : j + 1, :, :, :] = np.load(files[i + j])[0:c.data_height, 0:c.data_width, 0 : 1]  # 0:u,1:v
            #### 25-40N,125,140E ###
            data_single[j: j + 1, :, :, :] = np.load(files[i + j])[0 : c.data_height, c.data_width-1 : 241, 0 : 1]
        data[i, :, :, :, :] = data_single
    return data

def read_observe_data(files, batch_size, histlen, futulen):
    m = 1  # the num of input channel
    data = np.empty([batch_size, histlen + 1 + futulen, c.data_height, c.data_width, m]) #
    for i in range(batch_size):
        data_interpolate = np.empty([histlen + 1 + futulen, c.data_height, c.data_width, m])
        for j in range(histlen + 1 + futulen):
            data_single = np.load(files[i + j])[::-1, :, 0 : 1]  # 0:u,1:v  ###### shangxia fanzhuan ######
            #### 25-40N,110-125E ###
            #data_single1 = data_single[0:int((c.data_height+1)/2), 0:int((c.data_width+1)/2), 0:1] #lyq add 220816
            #### 25-40N,125,140E ###
            data_single1 = data_single[0 : int((c.data_height + 1) / 2), int((c.data_width + 1) / 2)-1 : 241, 0 : 1]

            data_interpolate[j : j + 1, :, :, 0] = cv2.resize(data_single1, (c.data_height, c.data_width)) # cv2.resize
        data[i, :, :, :, :] = data_interpolate
    return data

def read_data_o(files, batch_size, histlen, futulen):
    m = 1 # the num of input channel
    data = np.empty([batch_size, c.data_height, c.data_width, m * (histlen + 1 + futulen)])
    for i in range(batch_size):
        data_single = np.empty([c.data_height, c.data_width, m * (histlen + 1 + futulen)])
        for j in range(histlen + 1 + futulen):
            #MaxMinNormalization() # 0-1
            data_single[:, :, m * j : m * (j + 1)] = MaxMinNormalization(np.load(files[i + j])[:, :, 1: 2])
        data[i, :, :, :] = data_single
    return data

def read_data_r(files, batch_size, histlen, futulen):
    m = 1 # the num of input channel
    data = np.empty([batch_size, c.data_height, c.data_width, m * (histlen + 1 + futulen)])
    for i in range(batch_size):
        data_single = np.empty([c.data_height, c.data_width, m * (histlen + 1 + futulen)])
        for j in range(histlen + 1 + futulen):
            data_single[:, :, m * j : m * (j + 1)] = np.load(files[i + j])[:, :, 1 : 2]
        data[i, :, :, :] = data_single
    return data

def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.999, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def psnr_error(gen_videos,gt_videos):
    shape = tf.shape(gen_videos)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3] * shape[4])
    square_diff = tf.to_float(tf.square(gt_videos - gen_videos))
    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1,2,3,4])))
    return tf.reduce_mean(batch_errors)

def log10(t):
    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

