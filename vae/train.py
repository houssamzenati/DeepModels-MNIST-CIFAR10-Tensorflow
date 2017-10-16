import tensorflow as tf
import logging
import importlib
from data import utilities

logger = logging.getLogger("variational.autoencoder.train")

batch_size = 200
latent_dim = 20
intermediate_dim = 128

def train(dataset, except_class_to_ignore, stop=100000):
    ''' 
    Trains the autoencoder on all dataset except the class/digit considered
    anomalous.
    Args: 
            dataset (str): name of the dataset, mnist or cifar10
            except_class_to_ignore (int): int in range 0 to 10, is the class/digit
                                          on which the neural net is not trained
    '''

    conv_varautoencoder = importlib.import_module('vae.conv_varautoencoder_{}' \
                                                              .format(dataset))
    data = importlib.import_module('data.{}'.format(dataset))

    logger.warn("The variational autoencoder is training on {}, ignoring the class {}". \
        format(dataset, except_class_to_ignore))

    data_generator = map((lambda inp: (inp[0], inp[1])), utilities. \
          infinite_generator(data.get_train(except_class_to_ignore), batch_size))

    # Input batch
    input_images = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")

    # Build the network
    z_mean, z_stddev = conv_varautoencoder.encoder(input_images, latent_dim, 
                                           intermediate_dim, name='varencoder')  
    samples = tf.random_normal([batch_size,latent_dim], 0, 1, dtype=tf.float32)
    guessed_z = z_mean + (tf.exp(z_stddev) * samples)
    generated_images = conv_varautoencoder.decoder(guessed_z, latent_dim, 
                                            intermediate_dim, name='vardecoder')

    # Flatten for the losses
    generated_flat = tf.contrib.layers.flatten(generated_images)
    input_flat = tf.contrib.layers.flatten(input_images)

    # Compute KL divergence (latent loss)
    latent_loss = -.5 * tf.reduce_sum(1. + z_stddev - tf.pow(z_mean, 2) - tf.exp(z_stddev), 
                                      reduction_indices=1)
    # Compute reconstruction loss
    reconstruction_loss = -tf.reduce_sum(input_flat * tf.log(1e-10 + generated_flat)
                               + (1-input_flat) * tf.log(1e-10 + 1 - generated_flat), 1)

    # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_flat, labels=input_flat), reduction_indices=1)

    # Global loss
    global_loss = tf.reduce_mean(latent_loss + reconstruction_loss, name='cost_function')

    # Visu
    output_visu = tf.cast(generated_images * 255.0, tf.uint8)
    input_visu = tf.cast(input_images * 255.0, tf.uint8)

    # Add summaries to visualise output images and losses
    summary_autoencoder = tf.summary.merge([ \
        tf.summary.scalar('summary/loss', global_loss), \
        tf.summary.image('summary/image/input', input_visu, max_outputs=3), \
        tf.summary.image('summary/image/output', output_visu, max_outputs=3)])

    # Variable for training step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Training variables
    encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='varencoder')
    decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='vardecoder')
    training_variables = encoder_variables + decoder_variables

    encoder_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                 scope='varencoder')
    decoder_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                 scope='vardecoder')
    update_ops = encoder_update_ops + decoder_update_ops

    with tf.control_dependencies(update_ops):
    	# Training operation
    	training_step = tf.train. \
    	                      AdamOptimizer(learning_rate=0.0002, beta1=0.5). \
    	                      minimize(global_loss, global_step=global_step,
    	                               var_list=training_variables)

    # We disable automatic summaries here, because the automatic system assumes that
    # any time you run any part of the graph, you will be providing values for _all_
    # summaries:
    logdir = "vae/train_logs/{}/{}". \
            format(dataset,except_class_to_ignore)

    sv = tf.train.Supervisor(logdir=logdir, global_step=global_step,
                             save_summaries_secs=None, save_model_secs=120)

    batch = 0
    with sv.managed_session() as session:
        # Set up tensorboard logging:
        logwriter = tf.summary.FileWriter(logdir, session.graph)

        while not sv.should_stop() and batch < stop:
    	    if batch > 0 and batch % 100 == 0:
    	    	logger.info('Step {}.'.format(batch))
    	    inp, _ = next(data_generator)
    	    _, summ = session.run([training_step, summary_autoencoder], 
                                         feed_dict={input_images: inp})
    	    logwriter.add_summary(summ, batch)
    	    batch += 1 

def run(dataset, except_class_to_ignore):
    ''' 
    Runs the training process
    Args: 
            dataset (str): name of the dataset, mnist or cifar10
            except_class_to_ignore (int): int in range 0 to 10, is the class/digit
                                          on which the neural net is not trained
    '''
    train(dataset, except_class_to_ignore)