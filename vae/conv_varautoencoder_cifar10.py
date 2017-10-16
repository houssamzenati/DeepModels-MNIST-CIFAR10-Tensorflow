import tensorflow as tf

def encoder(input_images, latent_dim, intermediate_dim=0, reuse=False, name='varencoder'):
	with tf.variable_scope(name, reuse=reuse):

		xavier_initializer = tf.contrib.layers.xavier_initializer()

		# Convolution outputs [batch, 16, 16, 3]
		conv1 = tf.layers.conv2d(inputs=input_images,
	                             filters=3,
	                             kernel_size=4,
	                             strides=2,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)

		# Convolution outputs [batch, 8, 8, 64]
		conv2 = tf.layers.conv2d(inputs=conv1,
	                             filters=64,
	                             kernel_size=4,
	                             strides=2,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)

		# Convolution outputs [batch, 4, 4, 64]
		conv3 = tf.layers.conv2d(inputs=conv2,
	                             filters=64,
	                             kernel_size=4,
	                             strides=2,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)

		flat = tf.contrib.layers.flatten(conv3)

		z_mean = tf.layers.dense(flat, units=latent_dim, name='z_mean')
		z_log_var = tf.layers.dense(flat, units=latent_dim, name='z_log_var')


		return z_mean, z_log_var

def decoder(z_flat, latent_dim, intermediate_dim=0, reuse=False, name='vardecoder'):
	with tf.variable_scope(name, reuse=reuse):
		xavier_initializer = tf.contrib.layers.xavier_initializer()

		# Dense outputs [batch, 4, 4, 64]
		z_develop = tf.layers.dense(z_flat, units=4*4*64)
		net = tf.nn.relu(tf.reshape(z_develop, [-1, 4, 4, 64]))

		# Transposed convolution outputs [batch, 8, 8, 64]
		net = tf.layers.conv2d_transpose(inputs=net, 
										 filters=64,
										 kernel_size=4,
										 strides=2,
										 padding='same',
                                         kernel_initializer=xavier_initializer,
                                         activation=tf.nn.relu)

		# Transposed convolution outputs [batch, 16, 16, 64]
		net = tf.layers.conv2d_transpose(inputs=net, 
										 filters=64,
										 kernel_size=4,
										 strides=2,
										 padding='same',
                                         kernel_initializer=xavier_initializer,
                                         activation=tf.nn.relu)

		# Transposed convolution outputs [batch, 32, 32, 3]
		net = tf.layers.conv2d_transpose(inputs=net, 
										 filters=3,
										 kernel_size=4,
										 strides=2,
										 padding='same',
                                         kernel_initializer=xavier_initializer)
                                        

		net = tf.nn.sigmoid(net)

		return net