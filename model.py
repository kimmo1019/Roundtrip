import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

#the default is relu function
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

# class Encoder(object):
#     def __init__(self, input_dim, output_dim, feat_dim, name):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.feat_dim = feat_dim
#         self.name = name

#     def __call__(self, x, reuse=True):
#         with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
#             fc1 = tc.layers.fully_connected(
#                 x, 256,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 activation_fn=tf.identity
#             )
#             fc1 = leaky_relu(fc1)

#             fc2 = tc.layers.fully_connected(
#                 fc1, 256,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 activation_fn=tf.identity
#             )
#             fc2 = leaky_relu(fc2)

#             fc3 = tc.layers.fully_connected(fc2, self.output_dim, activation_fn=tf.identity)               
#             logits = fc3[:, self.feat_dim:]
#             y = tf.nn.softmax(logits)
#             return fc3[:, 0:self.feat_dim], y, logits

#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]


#mapping the latent variable to a label representation
class Transformer(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
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
            logits = tcl.fully_connected(
                fc, self.output_dim,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            y = tf.nn.softmax(logits)
            return y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

if __name__=='__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian,batch_jacobian

    print 1./(2*np.pi)**2 * 0.2 * np.exp(-0.8)
    print 1./((2*np.pi)**2) * 0.2 * 2**2 *np.exp(-0.8)
    x_dim=2
    y_dim=2
    N=2
    sd_y=1.0
    y_points = np.ones((2,2))
    y_points[1] = 2*np.ones(2)
    x_points_ = np.ones((2,2))
    x_points_[1] = 2*np.ones(2)
    y_points__ = np.ones((2,2))
    y_points__[1] = 4*np.ones(2)
    rt_error = np.sum((y_points-y_points__)**2,axis=1)
    #get jocobian matrix with shape (N, y_dim, x_dim)
    #jacob_mat = np.random.normal(size=(N,y_dim,x_dim))
    jacob_mat = np.zeros((2,2,2))
    jacob_mat[0] = 2*np.eye(2)
    jacob_mat[1] = 4*np.eye(2)
    jacob_mat_transpose = jacob_mat.transpose((0,2,1))
    #matrix A = G^T(x_)*G(x_) with shape (N, x_dim, x_dim)
    A = map(lambda x, y: np.dot(x,y), jacob_mat_transpose, jacob_mat)
    #vector b = grad_^T(G(x_))*(y-y__) with shape (N, x_dim)
    b = map(lambda x, y: np.dot(x,y), jacob_mat_transpose, y_points-y_points__)
    #covariant matrix in constructed multivariate Gaussian with shape (N, x_dim, x_dim)
    Sigma = map(lambda x: np.linalg.inv(np.eye(x_dim)+x/sd_y**2),A)
    Sigma_inv = map(lambda x: np.eye(x_dim)+x/sd_y**2,A)
    #mean vector in constructed multivariate Gaussian with shape (N, x_dim)
    mu = map(lambda x,y,z: x.dot(y/sd_y**2-z),Sigma,b,x_points_)
    #constant c(y) in the integral c(y) = l2_norm(x_)^2 + l2_norm(y-y__)^2/sigma**2-mu^T*Sigma*mu
    c_y = map(lambda x,y,z,w: np.sum(x**2)+y/sd_y**2-z.T.dot(w).dot(z), x_points_, rt_error, mu, Sigma_inv)
    py_est = map(lambda x,y: 1./(np.sqrt(2*np.pi)*sd_y)**y_dim * np.sqrt(np.linalg.det(x)) * np.exp(-0.5*y), Sigma, c_y)
    print len(py_est),py_est
    print rt_error[0]
    print A[0]
    print b[0]
    print mu[0]
    print Sigma[0]
    print Sigma_inv[0]
    print c_y[0]
    sys.exit()
    g_net = Generator_resnet(input_dim=5,output_dim = 10,name='g_net',nb_layers=3,nb_units=10)
    #g_net = Generator_PCN(3,2,10,2,'g_net',nb_layers=3,nb_units=64)
    x = tf.placeholder(tf.float32, [1, 2], name='x')
    #y_ = g_net(x,reuse=False)
    y_  = x**2
    t = tf.reduce_mean((x - y_)**2)
    J = jacobian(y_,x)
    J2 = batch_jacobian(y_,x)
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    N=2
    a=np.array([[1,2,3,4,5,6]])
    b=np.tile(a,(N,1)).astype('float32')
    t_ = sess.run(t,feed_dict={x:np.array([[2,3]])})
    print t_
    sys.exit()
    #print len(g_net.vars[0]),g_net.vars[0]
    #print len(g_net.vars[1]),g_net.vars[1]
    #print len(g_net.vars[2]),g_net.vars[2]
    g_net_vars_z1 = g_net.vars[0]
    g_net_vars_z2 = g_net.vars[1]
    g_net_vars_zc = g_net.vars[2]
    print len(g_net_vars_z2),len(g_net_vars_zc)
    w_pretrain = sess.run(g_net_vars_z2)
    loss_w = tf.add_n([2 * tf.nn.l2_loss(v[0]-v[1]) for v in zip(g_net_vars_z2,w_pretrain)])
    loss_c = tf.add_n([2 * tf.nn.l2_loss(v) for v in g_net_vars_zc])
    adam_g = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.9)
    g_optim = adam_g.minimize(loss_w+loss_c, var_list=g_net_vars_z2+g_net_vars_zc)
    sess.run(tf.variables_initializer(adam_g.variables()))
    #print len(g_net.vars[0]),g_net.vars[0]
    #print len(g_net.vars[1]),g_net.vars[1]
    #print len(g_net.vars[2]),g_net.vars[2]
    sess.run(tf.variables_initializer(adam_g.variables()))
    w_pretrain = sess.run(g_net_vars_z2)
    print len(g_net_vars_z2),len(g_net_vars_zc)
    loss_w = tf.add_n([2 * tf.nn.l2_loss(v[0]-v[1]) for v in zip(g_net_vars_z2,w_pretrain)])
    loss_c = tf.add_n([2 * tf.nn.l2_loss(v) for v in g_net_vars_zc])
    adam_g = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.9)
    g_optim = adam_g.minimize(loss_w+loss_c, var_list=g_net_vars_z2+g_net_vars_zc)
    sess.run(tf.variables_initializer(adam_g.variables()))
    # print len(g_net.vars[0]),g_net.vars[0]
    # print len(g_net.vars[1]),g_net.vars[1]
    # print len(g_net.vars[2]),g_net.vars[2]
    sess.run(tf.variables_initializer(adam_g.variables()))
    print len(g_net_vars_z2),len(g_net_vars_zc)
    sys.exit()
    c = tf.add_n([2 * tf.nn.l2_loss(v) for v in g_net.vars[1]])
    a = [sess.run(v) for v in g_net.vars[1]]
    print len(g_net.vars[1])
    sess.run(tf.global_variables_initializer())
    print len(g_net.vars[1])
    b = tf.reduce_mean([ tf.nn.l2_loss(v[0]-v[1]) for v in zip(g_net.vars[1],a)])
    print sess.run(b)
    print sess.run(c)