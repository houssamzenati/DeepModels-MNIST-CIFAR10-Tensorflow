import tensorflow as tf
import logging
import importlib
from data import utilities


logger = logging.getLogger("autoencoder.train")
batch_size = 500
nb_epochs = 50

def train(dataset, except_class_to_ignore, stop=100000):
    ''' 
    Trains the autoencoder on all dataset except the class/digit considered
    anomalous.
    Args: 
            dataset (str): name of the dataset, mnist or cifar10
            except_class_to_ignore (int): int in range 0 to 10, is the class/digit
                                          on which the neural net is not trained
    '''

    conv_autoencoder = importlib.import_module('ae.conv_autoencoder_{}'.format(dataset))
    data = importlib.import_module('data.{}'.format(dataset))

    logger.warn("The autoencoder is training on {}, ignoring the class {}". \
        format(dataset, except_class_to_ignore))

    data_generator = map((lambda inp: (inp[0], inp[1])), utilities. \
        infinite_generator(data.get_train(except_class_to_ignore), batch_size))

    # Input batch
    input_images = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")

    # We have different architecture for different codes 
    if dataset == 'mnist':
        encoder = conv_autoencoder.encoder(input_images, name='encoder')
        decoder = conv_autoencoder.decoder(encoder, name='decoder')

    if dataset == 'cifar10':
        encoder = conv_autoencoder.encoder(input_images, code_length=128, 
                                           height=32, width=32, name='encoder')
        decoder = conv_autoencoder.decoder(encoder, height=32, width=32, name='decoder')

    # Reconstruction L2 loss
    loss = tf.nn.l2_loss(input_images - decoder)

    # Visualisation
    output_visu = tf.cast(decoder * 255.0, tf.uint8)
    input_visu = tf.cast(input_images * 255.0, tf.uint8)

    # Add summaries to visualise output images and losses
    summary_autoencoder = tf.summary.merge([ \
        tf.summary.scalar('summary/loss', loss), \
        tf.summary.image('summary/image/input', input_visu, max_outputs=3), \
        tf.summary.image('summary/image/output', output_visu, max_outputs=3)])

    # Variable for training step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Training variables
    encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
    decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
    training_variables = encoder_variables + decoder_variables

    # Training operation
    training_step = tf.train. \
                          AdamOptimizer(learning_rate=0.0002, beta1=0.5). \
                          minimize(loss, global_step=global_step,
                                   var_list=training_variables)

    # We disable automatic summaries here, because the automatic system assumes that
    # any time you run any part of the graph, you will be providing values for _all_
    # summaries:
    logdir = "ae/train_logs/{}/{}". \
            format(dataset,except_class_to_ignore)

    sv = tf.train.Supervisor(logdir=logdir, global_step=global_step,
                             save_summaries_secs=None, save_model_secs=40)

    batch = 0
    with sv.managed_session() as session:
        # Set up tensorboard logging:
        logwriter = tf.summary.FileWriter(logdir, session.graph)

        while not sv.should_stop() and batch < stop:
    	    if batch > 0 and batch % 100 == 0:
    	    	logger.info('Step {}.'.format(batch))
    	    inp, _ = next(data_generator)
    	    _, summ = session.run([training_step, summary_autoencoder], \
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
        
