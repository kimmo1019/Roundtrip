import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    #return tf.maximum(tf.minimum(0.0, alpha * x), x)
    return tf.maximum(0.0, x)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x , y*tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] ,tf.shape(y)[3]])], 3)

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
            x = z[:,:self.input_dim]
            y = z[:,self.input_dim:]
            y.set_shape([None,10])
            yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            if self.dataset=="mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            x = conv_cond_concat(x,yb)
            conv = tcl.convolution2d(x, 32, [4,4],[2,2],
                activation_fn=tf.identity
                )
            #(bs, 14, 14, 32)
            conv = leaky_relu(conv)
            conv = conv_cond_concat(conv,yb)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, 64, [4,4],[2,2],
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
            #(bs, 7, 7, 32)
            #fc = tf.reshape(conv, [bs, -1])
            fc = tcl.flatten(conv)
            #(bs, 1568)
            fc = tf.concat([fc, y], axis=1)
            fc = tcl.fully_connected(
                fc, 1024,
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            
            fc = tf.concat([fc,y],axis=1)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator_img_ucond(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
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
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
            #(bs, 7, 7, 32)
            #fc = tf.reshape(conv, [bs, -1])
            fc = tcl.flatten(conv)
            #(bs, 1568)
            fc = tcl.fully_connected(
                fc, 1024,
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)
            
            output = tcl.fully_connected(
                fc, self.output_dim,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
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
            y = z[:,-10:]
            yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            fc = tcl.fully_connected(
                z, 1024,
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = leaky_relu(fc)
            fc = tf.concat([fc, y], 1)

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
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = leaky_relu(fc)
            fc = conv_cond_concat(fc,yb)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                activation_fn=tf.identity
            )
            #(bs,14,14,64)
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = leaky_relu(conv)
            if self.dataset=='mnist':
                output = tcl.convolution2d_transpose(
                    conv, 1, [4, 4], [2, 2],
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            elif self.dataset=='cifar10':
                output = tcl.convolution2d_transpose(
                    conv, 3, [4, 4], [2, 2],
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator_img_ucond(object):
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
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
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
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = leaky_relu(fc)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                activation_fn=tf.identity
            )
            #(bs,14,14,64)
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = leaky_relu(conv)
            if self.dataset=='mnist':
                output = tcl.convolution2d_transpose(
                    conv, 1, [4, 4], [2, 2],
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            elif self.dataset=='cifar10':
                output = tcl.convolution2d_transpose(
                    conv, 3, [4, 4], [2, 2],
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Encoder(object):
    def __init__(self, input_dim, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        # with tf.variable_scope(self.name) as vs:
        #     if reuse:
        #         vs.reuse_variables()
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
                fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
            return output[:, 0:self.feat_dim], y, logits

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
            
            # if self.cond:
            #     output = tcl.fully_connected(
            #         fc, self.output_dim+10, 
            #         activation_fn=tf.identity
            #         )
            #     logits = output[:, self.output_dim:]
            #     y = tf.nn.softmax(logits)
            #     return output[:,:self.output_dim], y, logits
            # else:
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

