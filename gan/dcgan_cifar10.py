# TensorFlow implementation of a DCGAN model that generates 32x32x3 images
# Originally written by Kingsley Kuan

import tensorflow as tf

def leaky_relu(features, alpha=0.2, name=None):
  return tf.maximum(alpha * features, features, name)

def generator(inputs, is_training=False, reuse=False, name='generator'):
    with tf.variable_scope(name, reuse=reuse):

        normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        
        # Expand input to [batch, 1, 1, channels]
        inputs = tf.expand_dims(tf.expand_dims(inputs, 1), 1)
        
        # Transposed convolution outputs [batch, 4, 4, 1024]
        net = tf.layers.conv2d_transpose(inputs, 
                                         filters=1024, 
                                         kernel_size=4, 
                                         padding='valid',
                                         kernel_initializer=normal_initializer,
                                         trainable=is_training,
                                         name='tconv1')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='tconv1/batch_normalization')

        net = tf.nn.relu(net, name='tconv1/relu')
        
        # Transposed convolution outputs [batch, 8, 8, 256]
        net = tf.layers.conv2d_transpose(net, 
                                         filters=256, 
                                         kernel_size=4, 
                                         strides=2, 
                                         padding='same',
                                         kernel_initializer=normal_initializer,
                                         trainable=is_training,
                                         name='tconv2')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='tconv2/batch_normalization')
        net = tf.nn.relu(net, name='tconv2/relu')
        
        # Transposed convolution outputs [batch, 16, 16, 64]
        net = tf.layers.conv2d_transpose(net, 
                                         filters=64, 
                                         kernel_size=4, 
                                         strides=2, 
                                         padding='same',
                                         kernel_initializer=normal_initializer,
                                         trainable=is_training,
                                         name='tconv3')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='tconv3/batch_normalization')
        net = tf.nn.relu(net, name='tconv3/relu')
        
        # Transposed convolution outputs [batch, 32, 32, 3]
        net = tf.layers.conv2d_transpose(net, 
                                         filters=3, 
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
        
        # Convolution outputs [batch, 16, 16, 64]
        net = tf.layers.conv2d(inputs, 
                               filters=64, 
                               kernel_size=4, 
                               strides=2, 
                               padding='same',
                               kernel_initializer=normal_initializer,
                               trainable=is_training,
                               name='conv1')

        net = leaky_relu(net, 0.2, name='conv1/leaky_relu')
        
        # Convolution outputs [batch, 8, 8, 256]
        net = tf.layers.conv2d(net, 
                               filters=256, 
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
        
        # Convolution outputs [batch, 4, 4, 1024]
        net = tf.layers.conv2d(net, 
                               filters=1024, 
                               kernel_size=4, 
                               strides=2, 
                               padding='same',
                               kernel_initializer=normal_initializer,
                               trainable=is_training,
                               name='conv3')

        net = tf.layers.batch_normalization(net, 
                                            training=is_training,
                                            name='conv3/batch_normalization')

        net = leaky_relu(net, 0.2, name='conv3/leaky_relu')
        
        # Convolution outputs [batch, 1, 1, 1]
        net = tf.layers.conv2d(net, 
                               filters=1, 
                               kernel_size=4, 
                               padding='valid',
                               kernel_initializer=normal_initializer,
                               trainable=is_training,
                               name='conv4')
        
        # Squeeze height and width dimensions
        net = tf.squeeze(net, [1, 2, 3])
        
        return net