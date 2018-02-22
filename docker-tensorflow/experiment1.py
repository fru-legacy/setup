""" Copyright 2018 Florian Rüberg

Summary:

Autoencoder (autoenc) is trained without limiting the network architecture. An auxiliary network is used, which tries
to include adversarial information in the input (modifier) and read that information (decoder) in the reconstruction.
Only by using abstraction should the autoencoder be able to minimize the passing of adversarial information.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf


EPOCH_SIZE = 20000
BATCH_SIZE = 4
MNIST = tf.contrib.learn.datasets.load_dataset('mnist')
DATA_DIMS = [28, 28, 1]
HIDDEN_INFO_SIZE = 12


def data_next_batch():
    batch = MNIST.train.next_batch(BATCH_SIZE)
    return np.reshape(batch[0], [-1] + DATA_DIMS)


def random_secret_batch():
    return np.random.uniform(0.2, 0.8, size=(BATCH_SIZE, HIDDEN_INFO_SIZE))


def build_generator(input, width=2, training=True, shape=[49, 4, 4]):

    with tf.variable_scope('autoencoder', reuse=tf.AUTO_REUSE):

        def relu(i):
            return tf.nn.relu(tf.layers.batch_normalization(i, training=training))

        def deconv(i, size):
            return tf.layers.conv2d_transpose(i, size, 5, strides=2, padding='SAME');

        model = tf.reshape(input, [-1] + shape)
        model = relu(tf.layers.dense(model, np.prod(shape)))
        model = relu(deconv(model, width * 4))
        model = relu(deconv(model, width * 2))
        model = tf.tanh(model)
        model = tf.layers.dense(tf.layers.flatten(model), np.prod(DATA_DIMS))
        model = tf.reshape(model, [-1] + DATA_DIMS)
        return model


def build_discriminator(input, width=6, training=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        def leaky_relu(i):
            return tf.nn.leaky_relu(tf.layers.batch_normalization(i, training=training))

        def conv(i, size):
            return tf.layers.conv2d(i, size, 5, strides=2, padding='SAME');

        model = leaky_relu(conv(input, width))
        model = leaky_relu(conv(model, width * 2))
        model = leaky_relu(conv(model, width * 4))
        model = leaky_relu(conv(model, width * 8))
        model = tf.layers.dense(tf.layers.flatten(model), 2)
        return model


def build_modified_autoencoder(input, secret):
    with tf.variable_scope('adversary/modified_input'):
        model = tf.layers.dense(secret, HIDDEN_INFO_SIZE*3, activation=tf.nn.leaky_relu)
        model = tf.layers.dense(model, 28*28, activation=tf.nn.leaky_relu)
        model = tf.reshape(model, [-1] + DATA_DIMS)
        model = tf.clip_by_value(model, 0, 0.2)
        model = tf.add(input, model)
    return build_generator(model)


def build_extractor(input):
    with tf.variable_scope('adversary/extract_secret'):
        model = tf.layers.conv2d(input, 12, 7, strides=4, activation=tf.nn.leaky_relu)
        model = tf.layers.dense(model, 32, activation=tf.nn.leaky_relu)
        model = tf.layers.flatten(model)
        model = tf.layers.dense(model, HIDDEN_INFO_SIZE, activation=tf.nn.relu)
        return model


autoencoder_input = tf.placeholder('float', shape=[None] + DATA_DIMS)
autoencoder = build_generator(autoencoder_input)

adversary_secret = tf.placeholder('float', shape=[None, HIDDEN_INFO_SIZE])
adversary_intermediate = build_modified_autoencoder(autoencoder_input, adversary_secret)
adversary_output = build_extractor(adversary_intermediate)

autoencoder_loss = tf.reduce_mean(tf.square(autoencoder - autoencoder_input))
adversary_loss = tf.reduce_mean(tf.square(adversary_secret - adversary_output))

discriminated_input = build_discriminator(autoencoder_input)
discriminated_autoencoder = build_discriminator(autoencoder)
discriminated_adversary = build_discriminator(adversary_intermediate)
discriminated_concat = tf.concat([discriminated_input, discriminated_autoencoder, discriminated_adversary], 0)

discriminator_help_target = tf.concat([tf.ones([BATCH_SIZE * 2, 1]), tf.zeros([BATCH_SIZE, 1])], 0)
discriminator_harm_target = np.ones([BATCH_SIZE * 3, 1]) / 3

discriminator_help_loss = tf.reduce_mean(tf.square(discriminated_concat - discriminator_help_target))
discriminator_harm_loss = tf.reduce_mean(tf.square(discriminated_concat - discriminator_harm_target))

adversary_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'adversary')
autoencoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'autoencoder')
discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

autoencoder_equality_train_step = tf.train.AdamOptimizer().minimize(autoencoder_loss, var_list=autoencoder_vars)

adversary_train_step = tf.train.AdamOptimizer().minimize(adversary_loss, var_list=adversary_vars)
autoencoder_train_step = tf.train.AdamOptimizer().minimize(discriminator_harm_loss, var_list=autoencoder_vars + discriminator_vars)
discriminator_train_step = tf.train.AdamOptimizer().minimize(discriminator_help_loss, var_list=discriminator_vars)

tf.summary.image('input', autoencoder_input, 1)
tf.summary.image('autoencoder', autoencoder, 1)
tf.summary.image('adversary', adversary_intermediate, 1)

tf.summary.scalar('autoencoder_loss', autoencoder_loss)
tf.summary.scalar('adversary_loss', adversary_loss)
tf.summary.scalar('discriminator_harm_loss', discriminator_harm_loss)
tf.summary.scalar('discriminator_help_loss', discriminator_help_loss)

summary = tf.summary.merge_all()


def train_autoencoder(epoch):
    _, _, log = sess.run([autoencoder_equality_train_step, autoencoder_train_step, summary], feed_dict={
        autoencoder_input: data_next_batch(),
        adversary_secret: random_secret_batch()
    })
    summary_writer.add_summary(log, epoch)


pause_discriminator = True
def train_discriminator(epoch):
    #steps  = [adversary_train_step, autoencoder_train_step, discriminator_train_step]
    steps = [autoencoder_train_step]
    global pause_discriminator
    print(pause_discriminator)
    #if(not pause_discriminator):
    steps += [discriminator_train_step]
    result = sess.run(steps + [summary, discriminator_help_loss], feed_dict={
        autoencoder_input: data_next_batch(),
        adversary_secret: random_secret_batch()
    })
    pause_discriminator = result[len(steps)+1] < 0.2
    summary_writer.add_summary(result[len(steps)], epoch)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('log-exp1')

    for epoch in range(400):
        train_autoencoder(epoch)
    for epoch in range(EPOCH_SIZE):
        train_discriminator(400 + epoch)
