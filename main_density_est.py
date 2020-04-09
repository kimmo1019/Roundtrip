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
import metric
import util

'''
Instructions: Roundtrip model for density estimation
    x,y - data drawn from base density and observation data (target density)
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping x space to y space
    H(.)  - generator network for mapping y space to x space
    Dx(.) - discriminator network in x space
    Dy(.) - discriminator network in y space
'''
class RoundtripModel(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, pool, batch_size, alpha, beta, sd_y, df, scale):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.pool = pool
        self.sd_y = sd_y
        self.df = df
        self.scale = scale
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x,reuse=False)
        self.J = batch_jacobian(self.y_, self.x)
        self.x_ = self.h_net(self.y,reuse=False)

        self.x__ = self.h_net(self.y_)
        self.y__ = self.g_net(self.x_)

        self.dy_ = self.dy_net(self.y_, reuse=False)
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
        self.fake_y = tf.placeholder(tf.float32, [None, self.y_dim], name='fake_y')
        
        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(self.y)

        self.d_fake_x = self.dx_net(self.fake_x)
        self.d_fake_y = self.dy_net(self.fake_y)

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
        self.graph_dir = 'graph/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}_sd={}_df={}_scale={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta, self.sd_y, self.df, self.scale)
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/density_est/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}_sd={}_df={}_scale={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta, self.sd_y, self.df, self.scale)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=500)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)


    def train(self, epochs, cv_epoch, patience):
        data_y_train = copy.copy(self.y_sampler.X_train)
        data_y_test = self.y_sampler.X_test
        data_y_val = self.y_sampler.X_val
        best_likelihood_val = -np.inf
        counter = 1
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        start_time = time.time()
        for epoch in range(epochs):
            np.random.shuffle(data_y_train)
            lr = 2e-4 #if epoch < epochs/2 else 1e-4 #*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = len(data_y_train) // self.batch_size
            for idx in range(batch_idxs):
                bx = self.x_sampler.get_batch(self.batch_size)
                by = data_y_train[self.batch_size*idx:self.batch_size*(idx+1)]
                #update G and get generated fake data
                fake_bx, fake_by, g_summary, _ = self.sess.run([self.x_,self.y_,self.g_merged_summary ,self.g_h_optim], feed_dict={self.x: bx, self.y: by, self.lr:lr})
                self.summary_writer.add_summary(g_summary,counter)
                #random choose one batch from the previous 50 batches
                [fake_bx,fake_by] = self.pool([fake_bx,fake_by])
                #update D
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim], feed_dict={self.x: bx, self.y: by, self.fake_x: fake_bx, self.fake_y: fake_by,self.lr:lr})
                self.summary_writer.add_summary(d_summary,counter)
                #quick test on a random batch data
                if counter % 100 == 0:
                    bx = self.x_sampler.train(batch_size)
                    by = self.y_sampler.train(batch_size)

                    g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, g_loss, \
                        h_loss, g_h_loss, fake_bx, fake_by = self.sess.run(
                        [self.g_loss_adv, self.h_loss_adv, self.l2_loss_x, self.l2_loss_y, \
                        self.g_loss, self.h_loss, self.g_h_loss, self.x_, self.y_],
                        feed_dict={self.x: bx, self.y: by}
                    )
                    dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                        feed_dict={self.x: bx, self.y: by, self.fake_x: fake_bx, self.fake_y: fake_by})

                    print('Epoch [%d] Iter [%d] Time [%5.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f]' %
                        (epoch, counter, time.time() - start_time, g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
                counter+=1

            if epoch == cv_epoch:
                best_sd, best_scale = self.model_selection()
            if epoch >= cv_epoch:
                py_est_val = self.estimate_py_with_IS(data_y_val,epoch,sd_y=best_sd,scale=best_scale,sample_size=20000,log=True,save=False)
                average_likelihood_val = np.mean(py_est_val)
                sd_likelihood_val = np.std(py_est_val)/np.sqrt(len(py_est_val))
                f=open('%s/val_likelihood.txt'%self.save_dir,'a+')
                f.write('%d\t%f\t%f\t%f\t%f\n'%(epoch,average_likelihood_val,sd_likelihood_val))
                f.close()
                if average_likelihood_val > best_likelihood_val:
                    best_likelihood_val = average_likelihood_val
                    wait=0
                    py_est_test = self.estimate_py_with_IS(data_y_test,epoch,sd_y=best_sd,scale=best_scale,sample_size=40000,log=True)
                    average_likelihood_test = np.mean(py_est_test)
                    sd_likelihood_test = np.std(py_est_test)/np.sqrt(len(py_est_test))
                    f=open('%s/test_likelihood.txt'%self.save_dir,'a+')
                    f.write('%d\t%f\t%f\t%f\t%f\n'%(epoch,average_likelihood_test,sd_likelihood_test))
                    f.close()
                    self.save(epoch)
                else:
                    wait+=1
                    if wait>patience or epoch+1==epochs:
                        print('Early stopping at %d with best sd:%f, best scale:%f, test average likelihood%f, test sd likelihood%f'%(epoch,best_sd,best_scale, average_likelihood_test,sd_likelihood_test))
                        sys.exit()
            

    #selection the best sd and scale 
    def model_selection(self,sample_size=10000):
        data_y_val = self.y_sampler.X_val
        sd_list = [0.05,0.1,0.5]
        scale_list = [0.005,0.01,0.1,0.5]
        records = []
        for sd in sd_list:
            for scale in scale_list:
                py_est = self.estimate_py_with_IS(data_y_val,0,sd_y=sd,scale=scale,sample_size=sample_size,log=True,use_ess=True,save=False)
                records.append([sd,scale,np.mean(py_est)])
        #sort according to the likelihood of validation set
        records.sort(key=lambda item:item[-1])
        best_sd, best_scale = records[-1][0],records[-1][1]
        return best_sd, best_scale

    #predict with y_=G(x)
    def predict_y(self, x, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_y_ = self.sess.run(self.y_, feed_dict={self.x:batch_x})
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

    #calculate gradient 
    def get_jacobian(self,x,bs=16):
        N = x.shape[0]
        jcob_pred = np.zeros(shape=(N, self.y_dim, self.x_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_J = self.sess.run(self.J, feed_dict={self.x:batch_x})
            jcob_pred[ind, :] = batch_J
        return jcob_pred

    #estimate pdf of y (e.g., p(y)) with importance sampling
    def estimate_py_with_IS(self,y_points,epoch,sd_y=0.45,scale=0.5,sample_size=20000,bs=1024,log=True,use_ess=True,save=True):
        np.random.seed(0)
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
            y_point = zip_list[1]
            y_points_ = self.predict_y(x_points)
            y_points_ = y_points_.astype('float64')
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
            batch_x_points_ = x_points_[ind, :]
            batch_x_points_sample_list = program_paral(sample_from_qx,batch_x_points_)
            batch_py_given_x_list = program_paral(py_given_x, zip(batch_x_points_sample_list, batch_y_points))
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
            np.savez('%s/py_est_at_epoch%d_sd%f_scale%f.npz'%(self.save_dir,epoch,sd_y,scale), py_est, y_points)

        return py_est

    #estimate pdf of y (e.g., p(y)) with closed form formulation
    def estimate_py_with_CF(self,y_points,epoch,sd_y=0.45,scale=0.5,sample_size=30000,log=True,save=True):
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
        y_points__ = self.predict_y(x_points_)
        rt_error = np.sum((y_points-y_points__)**2,axis=1)
        #get jocobian matrix with shape (N, y_dim, x_dim)
        self.J = batch_jacobian(self.y_, self.x)
        jacob_mat = self.get_jacobian(x_points_)
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
            np.savez('%s/py_est_at_epoch%d_sd%f_scale%f_cf.npz'%(self.save_dir,epoch,sd_y,scale), py_est, y_points)
        return py_est

    def save(self,epoch):

        checkpoint_dir = 'checkpoint/{}/{}'.format(self.data, self.timestamp)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'),global_step=epoch)

    def load(self, pre_trained = False, timestamp='',epoch=999):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_{}_{}}'.format(self.data, self.x_dim,self.y_dim, self.alpha, self.beta)
        else:
            if timestamp == '':
                print('Best Timestamp not provided. Abort !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint/{}/{}'.format(self.data, timestamp)

        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-%d'%epoch))
        print('Restored model weights.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='simutation_data')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--cv_epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--sd_y', type=float, default=0.5,help='standard deviation in density estimation')
    parser.add_argument('--df', type=float, default=1,help='degree of freedom of student t distribution')
    parser.add_argument('--scale', type=float, default=1,help='scale of student t distribution')
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='False')
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    epochs = args.epochs
    cv_epoch = args.cv_epoch
    patience = args.patience
    alpha = args.alpha
    beta = args.beta
    sd_y = args.sd_y
    df = args.df
    scale = args.scale
    timestamp = args.timestamp
    
    g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=512)
    h_net = model.Generator(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=10,nb_units=256)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
    dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=4,nb_units=256)
    
    xs = util.Gaussian_sampler(N=5000,mean=np.zeros(x_dim),sd=1.0)
    #ys = util.Gaussian_sampler(N=10000,mean=np.zeros(y_dim),sd=1.0)
    #ys = util.GMM_sampler(N=10000,n_components=5,dim=y_dim,sd=0.5)
    #ys = util.GMM_sampler(N=50000,n_components=15,dim=y_dim,sd=0.5)
    #ys = util.Multi_dis_sampler(N=20000,dim=y_dim)
    #ys = util.miniboone_sampler()
    #ys = util.power_sampler()
    #ys = util.gas_sampler()
    #ys = util.hepmass_sampler()
    #ys = util.UCI_sampler('datasets/YearPredictionMSD/data.npy')
    #ys = util.Outliner_sampler('datasets/Outliner/ForestCover/data.npz')
    #ys = util.Gaus_Uni_sampler(N=10000,mean=0,sd=1.0,norm_dim=1,uni_dim=y_dim-1)
    ys = util.GMM_indep_sampler(N=50000, sd=0.1, dim=y_dim, n_components=3, bound=1)
    #np.savez('data_indep_dim%d.npz'%y_dim,ys.X_train,ys.X_val,ys.X_test)
    #ys = util.GMM_indep_sampler(N=50000, sd=0.05, dim=x_dim, n_components=5, bound=1)
    pool = util.DataPool()
    ################ gaussian mixture with 2 components############
    # mean = np.array([[0.25, 0.25],[0.75, 0.75]])
    # cov1 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov2 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov = np.array([cov1,cov2])
    # weights = [0.4,0.6]
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov,weights=weights)

    ################ gaussian mixture with 4 components############
    # mean = 0.5*np.array([[1, 1],[-1, 1],[1, -1],[-1, -1]])
    # sd = 0.1
    # cov = np.array([(sd**2)*np.eye(mean.shape[-1]) for item in range(len(mean))])
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov)

    ################ gaussian mixture with 8 components in three dimensional space#####
    # mean = 0.75*np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,1],[-1,1,-1], \
    #     [1,-1,-1],[-1,-1,-1]])
    # sd = 0.05
    # cov = np.array([(sd**2)*np.eye(mean.shape[-1]) for item in range(len(mean))])
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov)

    ################ gaussian mixture with 2**n components in n dimensional space#####
    # linspace_list = 0.75*np.array([np.linspace(-1.,1.,2) for _ in range(y_dim)])
    # mesh_grids_list = np.meshgrid(*linspace_list)
    # mean = np.vstack([item.ravel() for item in mesh_grids_list]).T
    # sd = 0.05
    # cov = np.array([(sd**2)*np.eye(mean.shape[-1]) for item in range(len(mean))])
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov)
    ################ gaussian mixture with 8-10 components forming a circle############
    # n_components = 8
    # def cal_cov(theta,sx=1,sy=0.4**2):
    #     Scale = np.array([[sx, 0], [0, sy]])
    #     #theta = 0*np.pi/4.0
    #     c, s = np.cos(theta), np.sin(theta)
    #     Rot = np.array([[c, -s], [s, c]])
    #     T = Rot.dot(Scale)
    #     Cov = T.dot(T.T)
    #     return Cov
    # radius = 3
    # mean = np.array([[radius*math.cos(2*np.pi*idx/float(n_components)),radius*math.sin(2*np.pi*idx/float(n_components))] for idx in range(n_components)])
    # cov = np.array([cal_cov(2*np.pi*idx/float(n_components)) for idx in range(n_components)])
    # ys = util.GMM_sampler(N=20000,mean=mean,cov=cov)
    # print ys.X_train.shape, ys.X_val.shape, ys.X_test.shape
    # np.savez('data_gmm_8com.npz',ys.X_train,ys.X_val,ys.X_test)
    # sys.exit()

    ################ gaussian mixture with random rotation and scaling transformations ############
    # def get_gmm_param(max_scale=5):
    #     T = np.eye(y_dim)
    #     for i in range(y_dim-1):
    #         theta = np.random.uniform(-2*np.pi,2*np.pi)
    #         c, s = np.cos(theta), np.sin(theta)
    #         R = np.array([[c, -s], [s, c]])
    #         Rot = np.eye(y_dim)
    #         Rot[i:(i+2),i:(i+2)] = R
    #         T = T.dot(Rot)
    #     S = np.eye(y_dim)
    #     for i in range(y_dim):
    #         if np.random.rand() > 0.5:
    #             S[i,i] *= np.random.uniform(1,max_scale) 
    #         else:
    #             S[i,i] /= np.random.uniform(1,max_scale) 
    #     T = T.dot(S)
    #     Cov = T.dot(T.T)
    #     return Cov
    # n_components=10
    # np.random.seed(0)
    # mean = np.random.uniform(-5,5,(n_components,y_dim))
    # cov = np.array([get_gmm_param() for _ in range(n_components)])
    # ys = util.GMM_sampler(N=50000,mean=mean,cov=cov)
    # np.savez('data_gmm_trans_dim%d.npz'%y_dim,ys.X_train,ys.X_val,ys.X_test)
    ################ swiss roll##############
    #ys = util.Swiss_roll_sampler(N=20000)

    ################ gaussian mixture plus normal plus uniform############
    # mean = np.array([[0.25, 0.25, 0.25],[0.75, 0.75, 0.75]])
    # cov1 = np.array([[0.05**2, 0.03**2, 0],[0.03**2, 0.05**2, 0],[0,0,0.05**2]])
    # cov2 = (0.05*2)*np.eye(3)
    # cov = np.array([cov1,cov2])
    # weights = [0.5,0.5]
    # ys = util.GMM_Uni_sampler(N=10000,mean=mean,cov=cov,weights=weights)

    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, pool, batch_size, alpha, beta, sd_y, df, scale)

    if args.train == 'True':
        RTM.train(epochs=epochs,cv_epoch=cv_epoch,patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            epoch=199
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = epoch)
            data_y_test = RTM.y_sampler.X_test
            py,_,_ = RTM.estimate_py_with_IS(data_y_test,epoch,sd_y=0.5,scale=0.5,sample_size=30000,log=True)
            