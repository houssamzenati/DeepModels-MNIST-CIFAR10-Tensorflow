import tensorflow as tf

def encoder(input_images, code_length, height, width, reuse=False, name='encoder'):
    with tf.variable_scope(name, reuse=reuse):
    	# Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=input_images,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.tanh)

        # Convolutional output (flattened)
        conv_output = tf.contrib.layers.flatten(conv1)

        # Code layer
        code_layer = tf.layers.dense(inputs=conv_output,
                                     units=code_length,
                                     activation=tf.nn.tanh)
        
        # Code output layer
        code_output = tf.layers.dense(inputs=code_layer,
                                      units=(height - 2) * (width - 2) * 3,
                                      activation=tf.nn.tanh)

        return code_output

def decoder(code_output, height, width, reuse=False, name='decoder'):
    with tf.variable_scope(name, reuse=reuse):

    	# Deconvolution input
        deconv_input = tf.reshape(code_output, (-1, height - 2, width - 2, 3))

        # Deconvolution layer 1
        output_images = tf.layers.conv2d_transpose(inputs=deconv_input,
                                             filters=3,
                                             kernel_size=(3, 3),
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             activation=tf.sigmoid)

        return output_images