import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

#the default is relu function
def leaky_relu(x, alpha=0.2):
    #return tf.maximum(tf.minimum(0.0, alpha * x), x)
    return tf.maximum(0.0, x)

class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = tcl.batch_norm(fc)
                #fc = leaky_relu(fc)
                fc = tf.nn.tanh(fc)
            
            output = tcl.fully_connected(
                fc, 1, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#Discriminator for images, takes (bs,dim) as input
class Discriminator_img(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            x = z[:,:-10]
            y = z[:,-10:]
            if self.dataset=="mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            conv = tcl.convolution2d(x, 32, [4,4],[2,2],
                activation_fn=tf.identity
                )
            #(bs, 14, 14, 32)
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, 64, [4,4],[2,2],
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
            #(bs, 7, 7, 32)
            conv = tcl.flatten(conv)
            #(bs, 1568)
            fc = tcl.fully_connected(conv, 128, activation_fn=tf.identity)
            conv = tf.concat([conv,y],axis=1)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


#generator for images, G()
class Generator_img(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',is_training=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            fc = tcl.fully_connected(
                z, 1024,
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,is_training = self.is_training)#test problem may occur?
            fc = leaky_relu(fc)
            if self.dataset=='mnist':
                fc = tcl.fully_connected(
                    fc, 7*7*128,
                    activation_fn=tf.identity
                    )
                fc = tf.reshape(fc, tf.stack([bs, 7, 7, 128]))
            elif self.dataset=='cifar10':
                fc = tcl.fully_connected(
                    fc, 8*8*128,
                    activation_fn=tf.identity
                    )
                fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tc.layers.batch_norm(fc,is_training = self.is_training)
            fc = leaky_relu(fc)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                activation_fn=tf.identity
            )
            #(bs,14,14,64)
            conv = tc.layers.batch_norm(conv,is_training = self.is_training)
            conv = leaky_relu(conv)
            if self.dataset=='mnist':
                output = tcl.convolution2d_transpose(
                    conv, 1, [4, 4], [2, 2],
                    activation_fn=tf.nn.tanh
                )
                output = tf.reshape(output, [bs, -1])
            elif self.dataset=='cifar10':
                output = tcl.convolution2d_transpose(
                    conv, 3, [4, 4], [2, 2],
                    activation_fn=tf.nn.tanh
                )
                output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#encoder for images, H()
class Encoder_img(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',cond=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            if self.dataset=="mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            conv = tcl.convolution2d(x,64,[4,4],[2,2],
                activation_fn=tf.identity
                )
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, self.nb_units, [4,4],[2,2],
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
            conv = tcl.flatten(conv)
            fc = tcl.fully_connected(conv, 1024, activation_fn=tf.identity)
            
            if self.cond:
                output = tcl.fully_connected(
                    fc, self.output_dim+10, 
                    activation_fn=tf.identity
                    )
                logits = output[:, self.output_dim:]
                y = tf.nn.softmax(logits)
                return output[:,:self.output_dim], y, logits
            else:
                output = tcl.fully_connected(
                    fc, self.output_dim, 
                    activation_fn=tf.identity
                    )                
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]




if __name__=='__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian,batch_jacobian
    g_net = Generator_img(input_dim=10, output_dim=784, name='g_net', nb_layers=2,nb_units=256,dataset='mnist')
    d_net = Discriminator_img(input_dim=784+10, name='d_net', nb_layers=2,nb_units=256,dataset='mnist')
    x = tf.placeholder(tf.float32, [32, 10], name='x')
    y = g_net(x,reuse=False)
    z = tf.placeholder(tf.float32, [32, 784+10], name='x')
    dz = d_net(z,reuse=False)

    
    print z, dz

