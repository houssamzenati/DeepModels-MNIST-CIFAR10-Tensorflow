import tensorflow as tf

def encoder(input_images, latent_dim, intermediate_dim, reuse=False, name='varencoder'):
	with tf.variable_scope(name, reuse=reuse):

		xavier_initializer = tf.contrib.layers.xavier_initializer()

		# Convolution outputs [batch, 16, 16, 3]
		conv1 = tf.layers.conv2d(inputs=input_images,
	                             filters=1,
	                             kernel_size=2,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)


		# Convolution outputs [batch, 8, 8, 32]
		conv2 = tf.layers.conv2d(inputs=conv1,
	                             filters=64,
	                             kernel_size=2,
	                             strides=2,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)

		# Convolution outputs [batch, 4, 4, 64]
		conv3 = tf.layers.conv2d(inputs=conv2,
	                             filters=64,
	                             kernel_size=3,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)

		conv4 = tf.layers.conv2d(inputs=conv3,
	                             filters=64,
	                             kernel_size=3,
	                             padding='same',
	                             kernel_initializer=xavier_initializer,
	                             activation=tf.nn.relu)

		flat = tf.contrib.layers.flatten(conv4)

		hidden = tf.layers.dense(flat, units=intermediate_dim, activation=tf.nn.relu, name='dense')

		z_mean = tf.layers.dense(hidden, units=latent_dim, name='z_mean')
		z_log_var = tf.layers.dense(flat, units=latent_dim, name='z_log_var')

		return z_mean, z_log_var

def decoder(z_flat, latent_dim, intermediate_dim, reuse=False, name='vardecoder'):
	with tf.variable_scope(name, reuse=reuse):
		xavier_initializer = tf.contrib.layers.xavier_initializer()

		hidden_decoder = tf.layers.dense(z_flat, units=intermediate_dim, activation=tf.nn.relu, name='dense')

		# Dense outputs [batch, 4, 4, 64]
		z_develop = tf.layers.dense(hidden_decoder, units=14*14*64)
		net = tf.nn.relu(tf.reshape(z_develop, [-1, 14, 14, 64]))

		# Transposed convolution outputs [batch, 8, 8, 64]
		net = tf.layers.conv2d_transpose(inputs=net, 
										 filters=64,
										 kernel_size=3,
										 padding='same',
                                         kernel_initializer=xavier_initializer,
	                             		 activation=tf.nn.relu)

		# Transposed convolution outputs [batch, 16, 16, 32]
		net = tf.layers.conv2d_transpose(inputs=net, 
										 filters=64,
										 kernel_size=3,
										 padding='same',
                                         kernel_initializer=xavier_initializer,
	                                     activation=tf.nn.relu)

		# Transposed convolution outputs [batch, 32, 32, 3]
		net = tf.layers.conv2d_transpose(inputs=net, 
										 filters=64,
										 kernel_size=3,
										 strides=2,
										 padding='valid',
                                         kernel_initializer=xavier_initializer,
	                                     activation=tf.nn.relu)

		net = tf.layers.conv2d(inputs=net,
	                             filters=1,
	                             kernel_size=2,
	                             padding='valid',
	                             kernel_initializer=xavier_initializer)
                                        
		net = tf.sigmoid(net)

		return net