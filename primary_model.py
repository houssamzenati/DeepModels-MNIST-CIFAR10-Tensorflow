import numpy as np
import tensorflow as tf
import os
import os.path


h1_size = 150
h2_size = 350
img_height = 28
img_width = 28
img_size = img_height * img_width

def generator(z_dimension, batch_size):
	z_input = tf.truncated_normal([batch_size, z_dimension], mean=0, stddev=1, name='z_input')
	with tf.name_scope('g_fc_1'):
		w1 = tf.get_variable('g_w_fc1', [z_dimension, h1_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b1 = tf.get_variable('g_b_fc1', [h1_size], initializer=tf.zeros_initializer())
		fc1 = tf.matmul(z_input, w1) + b1
		act1 = tf.nn.relu(fc1)
		tf.summary.histogram('weights', w1)
		tf.summary.histogram('biaises', b1)
		tf.summary.histogram('activations', act1)
	with tf.name_scope('g_fc_2'):
		w2 = tf.get_variable('g_w_fc2', [h1_size, h2_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b2 = tf.get_variable('g_b_fc2', [h2_size], initializer=tf.zeros_initializer())
		fc2 = tf.matmul(act1, w2) + b2
		act2 = tf.nn.relu(fc2)
		tf.summary.histogram('weights', w2)
		tf.summary.histogram('biaises', b2)
		tf.summary.histogram('activations', act2)
	with tf.name_scope('g_fc_3'):
		w3 = tf.get_variable('g_w_fc3', [h2_size, img_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b3 = tf.get_variable('g_b_fc3', [img_size], initializer=tf.zeros_initializer())
		fc3 = tf.matmul(act2, w3) + b3
		g_output = tf.nn.tanh(fc3)
		tf.summary.histogram('weights', w3)
		tf.summary.histogram('biaises', b3)
		#tf.summary.histogram('activations', act2)
	return g_output

def discriminator(x_input, keep_prob):
	with tf.name_scope('d_fc_1'):
		w1 = tf.get_variable('d_w_fc1', [img_size, h2_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b1 = tf.get_variable('d_b_fc1', [h2_size], initializer=tf.zeros_initializer())
		fc1 = tf.matmul(x_input, w1) + b1
		act1 = tf.nn.relu(fc1)
		act1 = tf.nn.dropout(act1, keep_prob=keep_prob)
		tf.summary.histogram('weights', w1)
		tf.summary.histogram('biaises', b1)
		tf.summary.histogram('activations', act1)
	with tf.name_scope('d_fc_2'):
		w2 = tf.get_variable('d_w_fc2', [h2_size, h1_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b2 = tf.get_variable('d_b_fc2', [h1_size], initializer=tf.zeros_initializer())
		fc2 = tf.matmul(act1, w2) + b2
		act2 = tf.nn.relu(fc2)
		act2 = tf.nn.dropout(act2, keep_prob=keep_prob)
		tf.summary.histogram('weights', w2)
		tf.summary.histogram('biaises', b2)
		tf.summary.histogram('activations', act2)
	with tf.name_scope('d_fc_3'):
		w3 = tf.get_variable('d_w_fc3', [h1_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b3 = tf.get_variable('d_b_fc3', [1], initializer=tf.zeros_initializer())
		d_output = tf.matmul(act2, w3) + b3
		tf.summary.histogram('weights', w3)
		tf.summary.histogram('biaises', b3)
	return d_output
	
	