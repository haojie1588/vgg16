from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np

import tensorflow as tf

    # Build the AlexNet model
IMAGE_SIZE = 224    # input images size
KEEP_PROB = 0.5
is_training = False
batch_norm = False
wd=None
        # Parse input arguments into class variable
def _variable_on_cpu(name, shape, para):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
        name: name of the variable
        shape: list of ints
        para: parameter for initializer
    
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        if name == 'weights':
            # initializer = tf.truncated_normal_initializer(stddev=para, dtype=dtype)
            initializer = tf.contrib.layers.xavier_initializer(seed=1)
        else:
            initializer = tf.constant_initializer(para)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var
def VGG16( X,NUM_CLASSES,reuse=tf.AUTO_REUSE):
      # 1st Layer: Conv_1-2 (w ReLu) -> Pool
        conv1_1 = conv(X, 3, 3, 64, 1, 1, name='conv1_1', reuse=reuse)
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name='conv1_2', reuse=reuse)
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, name='pool1')

        # 2nd Layer: Conv_1-2 (w ReLu) -> Pool
        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', reuse=reuse)
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', reuse=reuse)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')
               
        # 3rd Layer: Conv_1-3 (w ReLu) -> Pool
        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', reuse=reuse)
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', reuse=reuse)
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', reuse=reuse)
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')

        # 4th Layer: Conv_1-3 (w ReLu) -> Pool
        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', reuse=reuse)
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', reuse=reuse)
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', reuse=reuse)
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')

        # 5th Layer: Conv_1-3 (w ReLu) -> Pool
        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', reuse=reuse)
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', reuse=reuse)
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', reuse=reuse)
        pool5 = max_pool(conv5_3, 2, 2, 2, 2, name='pool5')
        
        
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flat_shape = int(np.prod(pool5.get_shape()[1:]))
        flattened = tf.reshape(pool5, [-1, flat_shape])
        fc6 = fc(flattened, flat_shape, 4096, is_training,
                 name='fc6', reuse=reuse, batch_norm=batch_norm)
       

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, is_training, name='fc7',
                 reuse=reuse, batch_norm=batch_norm)
      
        fc8 = fc(fc7, 4096, NUM_CLASSES, is_training,
                 relu=False, name='fc8', reuse=reuse)

        return fc8
def conv(x, kernel_height, kernel_width, num_kernels, stride_y, stride_x, name,
         reuse=False, padding='SAME'):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
#    print(x.get_shape())

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_height, kernel_width,
                                               input_channels, num_kernels], 1e-1)
        biases = _variable_on_cpu('biases', [num_kernels], 0.0)

        # Apply convolution function
        conv = convolve(x, weights)

        # Add biases
        bias = tf.nn.bias_add(conv, biases)

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)
        act = tf.contrib.layers.batch_norm(relu, center=True, scale=True,
                                               trainable=True, is_training=is_training,
                                               decay=0.9,
                                                zero_debias_moving_mean=True,
                                               reuse=reuse, scope=scope)
        return act


def fc(x, num_in, num_out, is_training, name, reuse=False,
       relu=True, batch_norm=False):
    with tf.variable_scope(name, reuse=reuse) as scope:

        # Create tf variable for the weights and biases
        # weights = _variable_with_weight_decay('weights', [num_in, num_out], 1e-1, wd)
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)
        biases = _variable_on_cpu('biases', [num_out], 1.0)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if batch_norm:
            # Adds a Batch Normalization layer
            act = tf.contrib.layers.batch_norm(act, center=True, scale=True,
                                               trainable=True, is_training=is_training,
                                               decay=0.9,
                                                zero_debias_moving_mean=True,
                                               reuse=reuse, scope=scope)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, kernel_height, kernel_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
