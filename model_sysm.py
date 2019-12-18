import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, input_dim, name):
        self.input_dim = input_dim
        self.name = name

    def __call__(self,x,reuse=True):
        #with tf.variable_scope(name_or_scope=self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc1 = tc.layers.fully_connected(
                x, 64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)

            fc2 = tc.layers.fully_connected(
                fc1, 64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)

            fc3 = tc.layers.fully_connected(fc2, 1, 
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.identity
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, input_dim, output_dim, name):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(name_or_scope=self.name,reuse=tf.AUTO_REUSE) as vs:   
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc1 = tcl.fully_connected(
                x, 64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tcl.fully_connected(
                fc1, 64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)
            
            fc3 = tc.layers.fully_connected(
                fc2, self.output_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
