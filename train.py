import tensorflow as tf
import numpy as np
import primary_model as model
import os
import os.path

z_dimension = 100
batch_size = 256 
img_height = 28
img_width = 28
img_size = img_height * img_width
learning_rate = 1E-4
LOGDIR = '/tmp/mnist-gan2/'
keep_rate = 0.7
max_epoch = 20000

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def train_model():

    x_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_size], name="x_input")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    with tf.variable_scope('generator_model'):
    	x_generated = model.generator(z_dimension, batch_size)

    with tf.variable_scope('discriminator_model') as scope:
    	y_generated = model.discriminator(x_generated, keep_prob)
    	scope.reuse_variables()
     	y_data = model.discriminator(x_data, keep_prob)

    with tf.name_scope('generator_loss'):
    	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_generated, labels=tf.ones_like(y_generated)))

    with tf.name_scope('discriminator_loss'):
    	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_data, labels=tf.ones_like(y_data)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_generated, labels=tf.zeros_like(y_generated)))
        d_loss = d_loss_fake + d_loss_real

	tvars = tf.trainable_variables()

	d_vars = [var for var in tvars if 'd_' in var.name]
	g_vars = [var for var in tvars if 'g_' in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate)

    with tf.name_scope('training'):
        d_trainer = optimizer.minimize(d_loss, var_list=d_vars, name='discriminator_trainer')
        g_trainer = optimizer.minimize(g_loss, var_list=g_vars, name='generator_trainer')

    # Summary 

    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_real_loss', d_loss_real)
    tf.summary.scalar('Discriminator_fake_loss', d_loss_fake)
    tf.summary.scalar('Discriminator_total_loss', d_loss)

    tf.summary.image('Generated_images', x_generated, 10)
    x_data_reshaped = tf.reshape(x_data, shape=[-1, 28, 28, 1])
    tf.summary.image('data_images', x_data_reshaped, 10)
    merged_summary = tf.summary.merge_all()


    tf.reset_default_graph()
    sess = tf.Session()
    #saver = tf.train.Saver()


    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    


    sess.run(tf.global_variables_initializer())
    

    for i in range(max_epoch):
    	x_batch, _ = mnist.train.next_batch(batch_size) 
    	x_batch = 2 * x_batch.astype(np.float32) - 1 #set image dynamic to [-1,1]

    	

    	sess.run(d_trainer, feed_dict={x_data: x_batch, keep_prob: keep_rate})
    	sess.run(g_trainer, feed_dict={x_data: x_batch, keep_prob: keep_rate})

    	if i % 20 == 0:
    		summary = sess.run(merged_summary, feed_dict={x_data: x_batch, z_input: z_sample, keep_prob: keep_rate})
    		writer.add_summary(summary, i)
    

    	print('step %d' % i)
    		#saver.save(sess, os.path.join(LOGDIR, 'model.ckpt'), i)


