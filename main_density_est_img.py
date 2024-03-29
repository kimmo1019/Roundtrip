from __future__ import division
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
tf.set_random_seed(0)
import numpy as np
import random
import copy
import math
import util

'''
Instructions: Roundtrip model for conditional density estimation (e.g., images)
    x,y - data drawn from base density and observation data (target density)
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping x space to y space
    H(.)  - generator network for mapping y space to x space
    Dx(.) - discriminator network in x space (latent space)
    Dy(.) - discriminator network in y space (observation space)
'''
class RoundtripModel(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, data, pool, batch_size, nb_classes, alpha, beta, df, is_train):
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.beta = beta
        self.df = df
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot')
        self.x_combine = tf.concat([self.x,self.x_onehot],axis=1)
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x_combine,reuse=False)
        self.J = batch_jacobian(self.y_, self.x)
        self.x_ = self.h_net(self.y,reuse=False)

        self.x__ = self.h_net(self.y_)
        self.x_combine_ = tf.concat([self.x_, self.x_onehot],axis=1)
        self.y__ = self.g_net(self.x_combine_)

        self.dy_ = self.dy_net(tf.concat([self.y_, self.x_onehot],axis=1), reuse=False)
        self.dx_ = self.dx_net(self.x_, reuse=False)

        self.l1_loss_x = tf.reduce_mean(tf.abs(self.x - self.x__))
        self.l1_loss_y = tf.reduce_mean(tf.abs(self.y - self.y__))

        self.l2_loss_x = tf.reduce_mean((self.x - self.x__)**2)
        self.l2_loss_y = tf.reduce_mean((self.y - self.y__)**2)

        #(1-D(x))^2
        self.g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(self.dy_)  - self.dy_)**2)
        self.h_loss_adv = tf.reduce_mean((0.9*tf.ones_like(self.dx_) - self.dx_)**2)

        self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y


        self.fake_x = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x')
        self.fake_x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot')
        self.fake_x_combine = tf.concat([self.fake_x, self.fake_x_onehot],axis=1)
        self.fake_y = tf.placeholder(tf.float32, [None, self.y_dim], name='fake_y')
        
        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(tf.concat([self.y, self.x_onehot],axis=1))

        self.d_fake_x = self.dx_net(self.fake_x)
        self.d_fake_y = self.dy_net(tf.concat([self.fake_y, self.x_onehot],axis=1))

        #(1-D(x))^2
        self.dx_loss = (tf.reduce_mean((0.9*tf.ones_like(self.dx) - self.dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(self.d_fake_x) - self.d_fake_x)**2))/2.0
        self.dy_loss = (tf.reduce_mean((0.9*tf.ones_like(self.dy) - self.dy)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(self.d_fake_y) - self.d_fake_y)**2))/2.0
        self.d_loss = self.dx_loss + self.dy_loss

        #weight clipping
        self.clip_dx = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net.vars]
        self.clip_dy = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net.vars]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.g_loss_adv_summary = tf.summary.scalar('g_loss_adv',self.g_loss_adv)
        self.h_loss_adv_summary = tf.summary.scalar('h_loss_adv',self.h_loss_adv)
        self.l2_loss_x_summary = tf.summary.scalar('l2_loss_x',self.l2_loss_x)
        self.l2_loss_y_summary = tf.summary.scalar('l2_loss_y',self.l2_loss_y)
        self.dx_loss_summary = tf.summary.scalar('dx_loss',self.dx_loss)
        self.dy_loss_summary = tf.summary.scalar('dy_loss',self.dy_loss)
        self.g_merged_summary = tf.summary.merge([self.g_loss_adv_summary, self.h_loss_adv_summary,\
            self.l2_loss_x_summary,self.l2_loss_y_summary])
        self.d_merged_summary = tf.summary.merge([self.dx_loss_summary,self.dy_loss_summary])

        #graph path for tensorboard visualization
        self.graph_dir = 'graph/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.graph_dir) and is_train:
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=5000)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)


    def train(self, epochs, cv_epoch, patience):
        counter = 1
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        start_time = time.time()
        for epoch in range(epochs):
            lr = 2e-4 #if epoch < epochs/2 else 1e-4 #* float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = self.y_sampler.N // batch_size
            for idx in range(batch_idxs):
                bx = self.x_sampler.get_batch(batch_size)
                by, by_onehot = self.y_sampler.train(batch_size,label = True)
                #update G and get generated fake data
                fake_bx, fake_by, g_summary, _ = self.sess.run([self.x_, self.y_,self.g_merged_summary ,self.g_h_optim], feed_dict={self.x: bx,self.x_onehot:by_onehot, self.y: by, self.lr:lr})
                self.summary_writer.add_summary(g_summary,counter)
                #random choose one batch from the previous 50 batches
                [fake_bx,fake_by] = self.pool([fake_bx,fake_by])
                #update D
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim], feed_dict={self.x: bx,self.x_onehot:by_onehot, self.y: by, self.fake_x: fake_bx,self.fake_y: fake_by,self.lr:lr})
                self.summary_writer.add_summary(d_summary,counter)
                #quick test on a random batch data
                if counter % 100 == 0:
                    bx = self.x_sampler.train(batch_size)
                    by, by_onehot = self.y_sampler.train(batch_size,label = True)

                    g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, g_loss, \
                        h_loss, g_h_loss, fake_bx, fake_by = self.sess.run(
                        [self.g_loss_adv, self.h_loss_adv, self.l2_loss_x, self.l2_loss_y, \
                        self.g_loss, self.h_loss, self.g_h_loss, self.x_, self.y_],
                        feed_dict={self.x: bx,self.x_onehot:by_onehot, self.y: by}
                    )
                    dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                        feed_dict={self.x: bx,self.x_onehot:by_onehot, self.y: by, self.fake_x: fake_bx, self.fake_y: fake_by})

                    print('Epoch  [%d]  Iter [%d] Time [%.2f] g_loss_adv [%.4f] h_loss_adv [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f]' %
                        (epoch, counter, time.time() - start_time, g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
                counter+=1
            if (epoch+1) % 50 == 0 or epoch+1 == epochs:
                self.save(epoch)

    #predict with y_=G(x)
    def predict_y(self, x, x_d, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_d = x_d[ind,:]
            batch_y_ = self.sess.run(self.y_, feed_dict={self.x:batch_x,self.x_onehot:batch_x_d})
            y_pred[ind, :] = batch_y_
        return y_pred
    
    #predict with x_=H(y)
    def predict_x(self,y,bs=256):
        assert y.shape[-1] == self.y_dim
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_ = self.sess.run(self.x_, feed_dict={self.y:batch_y})
            x_pred[ind, :] = batch_x_
        return x_pred
    
    #calculate Jacobian matrix 
    def get_jacobian(self,x, x_d,bs=16):
        N = x.shape[0]
        jcob_pred = np.zeros(shape=(N, self.y_dim, self.x_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_d = x_d[ind,:]
            batch_J = self.sess.run(self.J, feed_dict={self.x:batch_x,self.x_onehot:batch_x_d})
            jcob_pred[ind, :] = batch_J
        return jcob_pred

    #estimate pdf of y (e.g., p(y)) with importance sampling
    def estimate_py_with_IS(self,y_points,onehot_label,epoch,sd_y=0.1,scale=0.01,sample_size=40000,bs=1024,log=True,save=True):
        np.random.seed(1024)
        from scipy.stats import t
        from multiprocessing.dummy import Pool as ThreadPool
        #multi-process to parallel the program
        def program_paral(func,param_list):
            pool = ThreadPool()
            results = pool.map(func,param_list)
            pool.close()
            pool.join()
            return results

        def py_given_x(zip_list):
            '''
            calculate p(y|x)
            x_points with shape (sample_size, x_dim)
            y_point wish shape (y_dim, )
            '''
            x_points = zip_list[0]
            labels = zip_list[1]
            y_point = zip_list[2]
            y_points_ = self.predict_y(x_points,labels)
            if log:
                return -self.y_dim*np.log((np.sqrt(2*np.pi)*sd_y))-(np.sum((y_point-y_points_)**2,axis=1))/(2.*sd_y**2)
            else:
                return 1. / ((np.sqrt(2*np.pi)*sd_y)**self.y_dim) * np.exp(-(np.sum((y_point-y_points_)**2,axis=1))/(2.*sd_y**2))

        def w_likelihood_ratio(zip_list):
            '''
            calculate w=px/py
            x_point with shape (x_dim, )
            x_points with shape (sample_size,x_dim)
            '''
            x_point = zip_list[0]
            x_points = zip_list[1]
            x_point = x_point.astype('float64')
            x_points = x_points.astype('float64')
            if log:
                log_qx = np.sum(t.logpdf(x_point-x_points,self.df,loc=0,scale=scale),axis=1)
                log_px = -self.x_dim*np.log(np.sqrt(2*np.pi))-(np.sum((x_points)**2,axis=1))/2.
                return log_px-log_qx
            else:
                qx =np.prod(t.pdf(x_point-x_points,self.df,loc=0,scale=scale),axis=1)
                px = 1. / (np.sqrt(2*np.pi)**self.x_dim) * np.exp(-(np.sum((x_points)**2,axis=1))/2.)
                return px / qx

        #sample a set of points given each x_point from importance distribution
        def sample_from_qx(x_point):
            '''
            multivariate student t distribution can be constructed from a multivariate Gaussian 
            one can also use t.rvs to sample (see the uncommented line) which is lower
            '''
            S = np.diag(scale**2 * np.ones(self.x_dim))
            z1 = np.random.chisquare(self.df, sample_size)/self.df
            z2 = np.random.multivariate_normal(np.zeros(self.x_dim),S,(sample_size,))
            return x_point + z2/np.sqrt(z1)[:,None]
            #return np.hstack([t.rvs(self.df, loc=value, scale=scale, size=(sample_size,1), random_state=None) for value in x_point])

        x_points_ = self.predict_x(y_points)
        N = len(y_points)
        py_given_x_list=[]
        w_likelihood_ratio_list=[]
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y_points = y_points[ind, :]
            batch_onehot_label = onehot_label[ind,:]
            batch_x_points_ = x_points_[ind, :]
            batch_x_points_sample_list = program_paral(sample_from_qx,batch_x_points_)
            batch_label_sample_list = [np.tile(item,(sample_size,1)) for item in batch_onehot_label]
            batch_py_given_x_list = program_paral(py_given_x, zip(batch_x_points_sample_list, batch_label_sample_list, batch_y_points))
            batch_w_likelihood_ratio_list = program_paral(w_likelihood_ratio, zip(batch_x_points_, batch_x_points_sample_list))
            py_given_x_list += batch_py_given_x_list
            w_likelihood_ratio_list += batch_w_likelihood_ratio_list


        #calculate p(y)=int(p(y|x)*p(x)dx)=int(p(y|x)*w(x)q(x)dx)=E(p(y|x)*w(x)) where x~q(x)
        if log:
            py_list = map(lambda x, y: x+y,py_given_x_list,w_likelihood_ratio_list)
            max_idx_list = [np.where(item==max(item))[0][0] for item in py_list]
            py_est = np.array([np.log(np.sum(np.exp(item[0]-item[0][item[1]])))-np.log(sample_size)+item[0][item[1]] for item in zip(py_list,max_idx_list)])
        else:
            py_list = map(lambda x, y: x*y,py_given_x_list,w_likelihood_ratio_list)
            py_est = np.array([np.mean(item) for item in py_list])
        if save:
            np.save('%s/py_est_at_epoch%d.npy'%(self.save_dir,epoch), py_est)
        return py_est

    #estimate pdf of y (e.g., p(y)) with Laplace approximation (closed-from)
    def estimate_py_with_CF(self,y_points,onehot_label,epoch,sd_y=0.1,log=True,save=True):
        from scipy.stats import t
        from multiprocessing.dummy import Pool as ThreadPool

        #multi-process to parallel the program
        def program_paral(func,param_list):
            pool = ThreadPool()
            results = pool.map(func,param_list)
            pool.close()
            pool.join()
            return results

        x_points_ = self.predict_x(y_points)
        y_points__ = self.predict_y(x_points_,onehot_label)
        rt_error = np.sum((y_points-y_points__)**2,axis=1)
        #get jocobian matrix with shape (N, y_dim, x_dim)
        jacob_mat = self.get_jacobian(x_points_,onehot_label)
        #jocobian matrix transpose with shape (N, x_dim, y_dim)
        jacob_mat_transpose = jacob_mat.transpose((0,2,1))
        #matrix A = G^T(x_)*G(x_) with shape (N, x_dim, x_dim)
        A = map(lambda x, y: np.dot(x,y), jacob_mat_transpose, jacob_mat)
        #vector b = grad_^T(G(x_))*(y-y__) with shape (N, x_dim)
        b = map(lambda x, y: np.dot(x,y), jacob_mat_transpose, y_points-y_points__)
        #covariant matrix in constructed multivariate Gaussian with shape (N, x_dim, x_dim)
        Sigma = map(lambda x: np.linalg.inv(np.eye(self.x_dim)+x/sd_y**2),A)
        Sigma_inv = map(lambda x: np.eye(self.x_dim)+x/sd_y**2,A)
        #mean vector in constructed multivariate Gaussian with shape (N, x_dim)
        mu = map(lambda x,y,z: x.dot(y/sd_y**2-z),Sigma,b,x_points_)
        #constant term c(y) in the integral c(y) = l2_norm(x_)^2 + l2_norm(y-y__)^2/sigma**2-mu^T*Sigma*mu
        c_y = map(lambda x,y,z,w: np.sum(x**2)+y/sd_y**2-z.T.dot(w).dot(z), x_points_, rt_error, mu, Sigma_inv)
        if log:
            py_est = map(lambda x,y:-self.y_dim*np.log(np.sqrt(2*np.pi)*sd_y)+0.5*np.log(np.linalg.det(x))-0.5*y, Sigma, c_y)
        else:
            py_est = map(lambda x,y: 1./(np.sqrt(2*np.pi)*sd_y)**self.y_dim* sd_y**self.y_dim *np.sqrt(np.linalg.det(x)) * np.exp(-0.5*y), Sigma, c_y)
        if save:
            np.save('%s/py_est_at_epoch%d.npy'%(self.save_dir,epoch), py_est)
        return py_est

    def save(self,epoch):

        checkpoint_dir = 'checkpoint/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'),global_step=epoch)

    def load(self, pre_trained = False, timestamp='',epoch=999):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}'.format(self.data)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-best'))
            print('Restored pre-trained model.')
        else:
            if timestamp == '':
                print('Best Timestamp not provided.')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-%d'%epoch))
                print('Restored model weights.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='model_img')
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--cv_epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--use_cv', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--df', type=float, default=1,help='degree of freedom of student t distribution')
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    nb_classes = args.K
    epochs = args.epochs
    cv_epoch = args.cv_epoch
    patience = args.patience
    alpha = args.alpha
    beta = args.beta
    df = args.df
    timestamp = args.timestamp
    use_cv = args.use_cv
    is_train = args.train

    if args.train:
        g_net = model.Generator_img(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=2,nb_units=256,dataset=data,is_training=True)
    else:
        g_net = model.Generator_img(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=2,nb_units=256,dataset=data,is_training=False)
    h_net = model.Encoder_img(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=2,nb_units=256,dataset=data,cond=True)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
    dy_net = model.Discriminator_img(input_dim=y_dim,name='dy_net',nb_layers=2,nb_units=128,dataset=data)
    pool = util.DataPool()

    xs = util.Gaussian_sampler(mean=np.zeros(x_dim),sd=1.0)
    if data=='mnist':
        best_sd, best_scale = 0.1, 0.01
        ys = util.mnist_sampler()
    elif data=='cifar10':
        best_sd, best_scale = 0.1, 0.01
        ys = util.cifar10_sampler()
    else:
        print("Wrong data name!")
        sys.exit()


    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, data, pool, batch_size, nb_classes, alpha, beta, df, is_train)

    if args.train:
        RTM.train(epochs=epochs,cv_epoch=cv_epoch,patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = epochs-1)

