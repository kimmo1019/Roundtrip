import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)
    #return tf.maximum(0.0, x)
    #return tf.nn.tanh(x)
    #return tf.nn.elu(x)



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

#fcn
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



class Generator_resnet(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def residual_block(self, x, dim):
        e = tcl.fully_connected(x, self.nb_units, activation_fn=tf.identity)
        e = leaky_relu(e)
        e = tcl.fully_connected(x, dim, activation_fn=tf.identity)
        e = leaky_relu(e)
        return x+e
        
    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                z, self.nb_units/2,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = self.residual_block(fc,self.nb_units)

            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)   
            fc = tcl.fully_connected(
                z, self.nb_units/2,
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


class Generator_res(object):#skip connection
    def __init__(self, input_dim, label_dim, output_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:   
        z_latent = z[:,:self.input_dim]    
        z_label = z[:,self.input_dim:]    
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
                fc = tf.concat([fc,z_label],axis=1)
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)
            #fc = tf.concat([fc,z_label],axis=1)
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

class Generator_Bayes(object):#y1,y2 = f(x1,x2) where p(y1|x1,x2) = p(y1|x1) 
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2,nb_units=256,constrain=False):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.constrain = constrain

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            z1 = z[:,:self.input_dim1]
            z2 = z[:,self.input_dim1:]

            fc1 = tcl.fully_connected(
                z1, self.nb_units,
                activation_fn=tf.identity,
                scope='z1_0' 
                )
            fc1 = leaky_relu(fc1)

            fc2 = tcl.fully_connected(
                z, self.nb_units,
                activation_fn=tf.identity,
                scope='z2_0'
                )
            fc2 = leaky_relu(fc2)     

            for i in range(self.nb_layers-1):
                z = fc1
                fc1 = tcl.fully_connected(
                    fc1, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z1_%d'%(i+1)
                    )
                fc1 = leaky_relu(fc1)

                fc2 = tf.concat([z,fc2],axis=1)
                fc2 = tcl.fully_connected(
                    fc2, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z2_%d'%(i+1)
                    )
                fc2 = leaky_relu(fc2)
            
            output1 = tcl.fully_connected(
                fc1, self.output_dim1,
                activation_fn=tf.identity,
                scope='z1_last'
                )
            fc2 = tf.concat([fc1,fc2],axis=1)
            output2 = tcl.fully_connected(
                fc2, self.output_dim2,
                activation_fn=tf.identity,
                scope='z2_last'
                )
            if self.constrain:
                output2_phi = output2[:,1:2]
                output2_sigma2 = output2[:,2:3]
                output2_nu = output2[:,3:4]
                output2_phi = tf.tanh(output2_phi)
                output2_sigma2 = tf.abs(output2_sigma2)
                #output2_nu = tf.abs(output2_nu)
                output2 = tf.concat([output2[:,0:1],output2_phi,output2_sigma2,output2_nu,output2[:,-2:]],axis=1)              
            return [output1,output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf.global_variables() if self.name+'/z1' in var.name]
        vars_z2 = [var for var in tf.global_variables() if self.name+'/z2' in var.name]
        all_vars = [var for var in tf.global_variables() if self.name in var.name]
        return [vars_z1,vars_z2,all_vars]

class Generator_PCN(object):#partially connected network, z1<--f1(z1), z2<--f2(z1,z2)
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2,nb_units=256):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            z1 = z[:,:self.input_dim1]
            z2 = z[:,self.input_dim1:]

            fc1 = tcl.fully_connected(
                z1, self.nb_units,
                activation_fn=tf.identity,
                scope='z1_0' 
                )
            fc1 = leaky_relu(fc1)
            #cross connections
            fc_cross = tcl.fully_connected(
                z1, self.nb_units,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=None,
                activation_fn=tf.identity,
                scope='zc_0'
                )
            fc_cross = leaky_relu(fc_cross)     

            fc2 = tcl.fully_connected(
                z2, self.nb_units,
                activation_fn=tf.identity,
                scope='z2_0'
                )
            fc2 = leaky_relu(fc2)              
            fc2 = tf.add(fc2,fc_cross)
            #fc2 = tf.concat([fc2,fc_cross],axis=1)
            
            for i in range(self.nb_layers-1):
                z = fc1
                fc1 = tcl.fully_connected(
                    fc1, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z1_%d'%(i+1)
                    )
                fc1 = leaky_relu(fc1)

                #cross connection
                fc_cross = tcl.fully_connected(
                    z, self.nb_units,
                    activation_fn=tf.identity,
                    weights_initializer=tf.zeros_initializer(),
                    biases_initializer=None,
                    scope='zc_%d'%(i+1)
                    )
                fc_cross = leaky_relu(fc_cross)

                fc2 = tcl.fully_connected(
                    fc2, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z2_%d'%(i+1)
                    )
                fc2 = leaky_relu(fc2)
                fc2 = tf.add(fc2,fc_cross)
                #fc2 = tf.concat([fc2,fc_cross],axis=1)

            output1 = tcl.fully_connected(
                fc1, self.output_dim1,
                activation_fn=tf.identity,
                scope='z1_last'
                )
            #cross connection
            output_cross = tcl.fully_connected(
                fc1, self.output_dim2,
                activation_fn=tf.identity,
                weights_initializer=tf.zeros_initializer(),
                scope='zc_last'
                )           
            
            output2 = tcl.fully_connected(
                fc2, self.output_dim2,
                activation_fn=tf.identity,
                scope='z2_last'
                )
            output2 = tf.add(output2,output_cross)        
            return [output1,output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf.global_variables() if self.name+'/z1' in var.name]
        vars_z2 = [var for var in tf.global_variables() if self.name+'/z2' in var.name]
        vars_zc = [var for var in tf.global_variables() if self.name+'/zc' in var.name]
        return [vars_z1,vars_z2,vars_zc]

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


if __name__=='__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian,batch_jacobian
