# TensorFlow implementation of a DCGAN model for MNIST

import tensorflow as tf

def leaky_relu(features, alpha=0.2, name=None):
  return tf.maximum(alpha * features, features, name)

def generator(inputs, is_training=False, reuse=False, name='generator'):
    with tf.variable_scope(name, reuse=reuse):

        normal_initializer = tf.random_normal_initializer(mean=0.0, 
                                                          stddev=0.02)

        # Fully Connected Layer 1: outputs [batch, 1024]
        net = tf.layers.dense(inputs, 
                              units=1024, 
                              kernel_initializer=normal_initializer, 
                              trainable=is_training, 
                              name='fc1')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='fc1/batch_normalization')

        net = tf.nn.relu(net, name='fc1/relu')

        # Fully Connected Layer 2: outputs [batch, 8, 8, 128]
        net = tf.layers.dense(net, 
                              units=7*7*128, 
                              kernel_initializer=normal_initializer, 
                              trainable=is_training, 
                              name='fc2')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='fc2/batch_normalization')

        net = tf.nn.relu(net, name='fc2/relu')

        net = tf.reshape(net, [-1, 7, 7, 128])

        # Transposed convolution outputs [batch, 14, 14, 64]
        net = tf.layers.conv2d_transpose(net, 
                                         filters=64, 
                                         kernel_size=4,
                                         strides= 2, 
                                         padding='same',
                                         kernel_initializer=normal_initializer,
                                         trainable=is_training,
                                         name='tconv3')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='tconv3/batch_normalization')

        net = tf.nn.relu(net, name='tconv3/relu')
        
        # Transposed convolution outputs [batch, 28, 28, 1]
        net = tf.layers.conv2d_transpose(net, 
                                         filters=1, 
                                         kernel_size=4, 
                                         strides=2, 
                                         padding='same',
                                         kernel_initializer=normal_initializer,
                                         trainable=is_training,
                                         name='tconv4')

        net = tf.tanh(net, name='tconv4/tanh')
        
        return net

def discriminator(inputs, is_training=False, reuse=False, name='discriminator'):
    with tf.variable_scope(name, reuse=reuse):

        normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        
        # Convolution outputs [batch, 14, 14, 64]
        net = tf.layers.conv2d(inputs, 
                               filters=64, 
                               kernel_size=4, 
                               strides=2, 
                               padding='same',
                               kernel_initializer=normal_initializer,
                               trainable=is_training,
                               name='conv1')

        net = leaky_relu(net, 0.2, name='conv1/leaky_relu')
        
        # Convolution outputs [batch, 7, 7, 256]
        net = tf.layers.conv2d(net, 
                               filters=64, 
                               kernel_size=4, 
                               strides=2, 
                               padding='same',
                               kernel_initializer=normal_initializer,
                               trainable=is_training,
                               name='conv2')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='conv2/batch_normalization')

        net = leaky_relu(net, 0.2, name='conv2/leaky_relu')
        
        # Fully connected 3 outputs [batch, 1024]
        net = tf.reshape(net, [-1, 7*7*64])
        net = tf.layers.dense(net, 
                              units=1024, 
                              kernel_initializer=normal_initializer, 
                              trainable=is_training, 
                              name='fc3')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='conv3/batch_normalization')

        net = leaky_relu(net, 0.2, name='fc3/leaky_relu')
        
        # Fully connected 4 outputs [batch, 1]
        net = tf.layers.dense(net, 
                              units=1, 
                              kernel_initializer=normal_initializer, 
                              trainable=is_training, 
                              name='fc4')
        
        net = tf.tanh(net, name='fc4/tanh')

        net = tf.squeeze(net)

        return net