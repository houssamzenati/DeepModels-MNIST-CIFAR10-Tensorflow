import tensorflow as tf
from keras.layers import UpSampling2D

def encoder(input_images, reuse=False, name='encoder'):
    with tf.variable_scope(name, reuse=reuse):
    	# Convolutional layer 1
        net = tf.layers.conv2d(inputs=input_images,
                                 filters=16,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='e_conv1')

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], 
                             strides=[1, 2, 2, 1], padding='SAME', name='maxpool_1')

        net = tf.layers.conv2d(inputs=net,
                                 filters=8,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='e_conv2')

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], 
                             strides=[1, 2, 2, 1], padding='SAME', name='maxpool_2')

        net = tf.layers.conv2d(inputs=net,
                                 filters=8,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='e_conv3')

        code_output = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], 
                             strides=[1, 2, 2, 1], padding='SAME', name='maxpool_3')

        return code_output

def decoder(code_output, reuse=False, name='decoder'):
    with tf.variable_scope(name, reuse=reuse):

        net = tf.layers.conv2d(inputs=code_output,
                                 filters=8,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='d_conv1')

        net = UpSampling2D((2, 2))(net)

        net = tf.layers.conv2d(inputs=net,
                                 filters=8,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='d_conv2')

        net = UpSampling2D((2, 2))(net)

        net = tf.layers.conv2d(inputs=net,
                                 filters=16,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 name='d_conv3')

        net = UpSampling2D((2, 2))(net)

        output_images = tf.layers.conv2d(inputs=net,
                                 filters=1,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='same',
                                 activation=tf.sigmoid,
                                 name='d_conv4')

        return output_images