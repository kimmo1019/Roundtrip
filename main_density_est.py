import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
import math
import metric
import util
from functools import reduce

tf.set_random_seed(0)

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
        self.use_L1=False# l2 is better than l1
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        tf.reset_default_graph()
        ##########################################################################
        '''
            x,y - two data distributions
            y_  - learned distribution by G(.), namely y_=G(x)
            x_  - learned distribution by H(.), namely x_=H(y)
            y__ - reconstructed distribution, y__ = G(H(y))
            x__ - reconstructed distribution, x__ = H(G(y))
        '''
        ##########################################################################

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x,reuse=False)
        self.x_ = self.h_net(self.y,reuse=False)

        self.x__ = self.h_net(self.y_)
        self.y__ = self.g_net(self.x_)

        self.dy_ = self.dy_net(self.y_, reuse=False)
        self.dx_ = self.dx_net(self.x_, reuse=False)

        self.l1_loss_x = tf.reduce_mean(tf.abs(self.x - self.x__))
        self.l1_loss_y = tf.reduce_mean(tf.abs(self.y - self.y__))

        self.l2_loss_x = tf.reduce_mean((self.x - self.x__)**2)
        self.l2_loss_y = tf.reduce_mean((self.y - self.y__)**2)

        #-D(x)
        #self.g_loss_adv = -tf.reduce_mean(self.dy_)
        #self.h_loss_adv = -tf.reduce_mean(self.dx_)
        #(1-D(x))^2
        self.g_loss_adv = tf.reduce_mean((tf.ones_like(self.dy_)  - self.dy_)**2)
        self.h_loss_adv = tf.reduce_mean((tf.ones_like(self.dx_) - self.dx_)**2)
        #cross_entropy
        #self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        #self.h_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_)))
        if self.use_L1:
            self.g_loss = self.g_loss_adv + self.alpha*self.l1_loss_x + self.beta*self.l1_loss_y
            self.h_loss = self.h_loss_adv + self.alpha*self.l1_loss_x + self.beta*self.l1_loss_y
            self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*self.l1_loss_x + self.beta*self.l1_loss_y
        else:
            self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
            self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
            self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y


        self.fake_x = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x')
        self.fake_y = tf.placeholder(tf.float32, [None, self.y_dim], name='fake_y')
        
        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(self.y)

        self.d_fake_x = self.dx_net(self.fake_x)
        self.d_fake_y = self.dy_net(self.fake_y)

        #-D(x)
        #self.dx_loss = tf.reduce_mean(self.dx_) - tf.reduce_mean(self.dx)
        #self.dy_loss = tf.reduce_mean(self.dy_) - tf.reduce_mean(self.dy)
        #(1-D(x))^2
        self.dx_loss = (tf.reduce_mean((tf.ones_like(self.dx) - self.dx)**2) \
                +tf.reduce_mean((tf.zeros_like(self.d_fake_x) - self.d_fake_x)**2))/2.0
        self.dy_loss = (tf.reduce_mean((tf.ones_like(self.dy) - self.dy)**2) \
                +tf.reduce_mean((tf.zeros_like(self.d_fake_y) - self.d_fake_y)**2))/2.0
        #log(D(x))
        # self.dx_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx, labels=tf.ones_like(self.dx))) \
        #         +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_x, labels=tf.zeros_like(self.d_fake_x))))/2.0
        # self.dy_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy, labels=tf.ones_like(self.dy))) \
        #         +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_y, labels=tf.zeros_like(self.d_fake_y))))/2.0

        self.d_loss = self.dx_loss + self.dy_loss

        self.clip_dx = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net.vars]
        self.clip_dy = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net.vars]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)

        # self.dx_adam = tf.train.RMSPropOptimizer(learning_rate=1e-3) \
        #         .minimize(self.dx_loss, var_list=self.dx_net.vars)
        # self.dy_adam = tf.train.RMSPropOptimizer(learning_rate=1e-3) \
        #         .minimize(self.dy_loss, var_list=self.dy_net.vars)
        # self.g_adam = tf.train.RMSPropOptimizer(learning_rate=1e-3) \
        #         .minimize(self.g_loss, var_list=self.g_net.vars)
        # self.h_adam = tf.train.RMSPropOptimizer(learning_rate=1e-3) \
        #         .minimize(self.h_loss, var_list=self.h_net.vars)
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        # self.g_loss_summary = tf.summary.scalar('g_loss',self.g_loss)
        # self.h_loss_summary = tf.summary.scalar('h_loss',self.h_loss)
        # self.g_h_loss_summary = tf.summary.scalar('g_h_loss',self.g_h_loss)
        # self.dx_loss_summary = tf.summary.scalar('dx_loss',self.dx_loss)
        # self.dy_loss_summary = tf.summary.scalar('dy_loss',self.dy_loss)
        # self.d_loss_summary = tf.summary.scalar('d_loss',self.d_loss)
        # self.merged_summary = tf.summary.merge([self.g_loss_summary, self.g_loss_summary,self.g_h_loss_summary,\
        #     self.dx_loss_summary,self.dy_loss_summary,self.d_loss_summary])
        # graph_dir = 'graph/{}_{}_{}_{}_{}'.format(self.timestamp,self.x_dim, self.y_dim, self.alpha, self.beta)
        # if not os.path.exists(graph_dir):
        #     os.makedirs(graph_dir)
        # self.summary_writer=tf.summary.FileWriter(graph_dir,graph=tf.get_default_graph())
        save_dir = 'data/density_est/density_est_{}_x_dim={}_y_dim={}_alpha={}_beta={}_sd={}_df={}_scale={}'.format(self.timestamp,self.x_dim, self.y_dim, self.alpha, self.beta, self.sd_y, self.df, self.scale)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, epochs=3000):
        #data_x, label_x = self.x_sampler.load_all()
        data_y, label_y = self.y_sampler.load_all()
        #data_x = np.array(data_x,dtype='float32')
        data_y = np.array(data_y,dtype='float32')
        batch_size = self.batch_size
        counter = 1
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for epoch in range(epochs):
            #np.random.shuffle(data_x)
            np.random.shuffle(data_y)
            lr = 2e-4 if epoch < epochs/2 else 2e-4*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = len(data_y) // batch_size
            #lr decay, add later
            for idx in range(batch_idxs):
                #bx = data_x[batch_size*idx:batch_size*(idx+1)]
                bx = self.x_sampler.get_batch(batch_size)
                #bx = self.x_sampler.train(batch_size)
                by = data_y[batch_size*idx:batch_size*(idx+1)]
                #update G and get generated fake data
                fake_bx, fake_by, _ = self.sess.run([self.x_,self.y_,self.g_h_optim], feed_dict={self.x: bx, self.y: by, self.lr:lr})
                #random choose one batch from the previous 50 batches,flash back
                [fake_bx,fake_by] = self.pool([fake_bx,fake_by])

                #update D
                self.sess.run(self.d_optim, feed_dict={self.x: bx, self.y: by, self.fake_x: fake_bx, self.fake_y: fake_by,self.lr:lr})
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

                    print('Iter [%8d] Time [%5.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f]' %
                        (counter, time.time() - start_time, g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
                counter+=1
                
                # summary = self.sess.run(self.merged_summary,feed_dict={self.x:bx,self.y:by})
                # self.summary_writer.add_summary(summary, counter)

            if (epoch+1) % 500 == 0:
                self.evaluate(epoch)
                y_mean = self.y_sampler.mean
                y_idx = np.random.choice(y_mean.shape[0], size=1000, replace=True)
                y_val = np.array([np.random.normal(y_mean[i],scale=0.5) for i in y_idx],dtype='float64')
                py_est = self.estimate_py_with_IS(y_val,epoch,sd_y=self.sd_y,sample_size=30000)
                #save model weights
                self.save(epoch)

    #predict with y_=G(x)
    def predict_y(self, x, bs=64):
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
    def predict_x(self,y,bs=64):
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

    #estimate pdf of y (e.g., p(y)) with importance sampling
    #parallel with each y_point
    def estimate_py_with_IS(self,y_points,epoch,sd_y=0.5,sample_size=30000,save=True):
        from scipy.stats import t
        from multiprocessing.dummy import Pool as ThreadPool
        t0=time.time()
        x_points_ = self.predict_x(y_points)
        x_points_sample_list = [np.hstack([t.rvs(self.df, loc=value, scale=self.scale, size=(sample_size,1), random_state=None) for value in point]) for point in x_points_]
        y_points_sample_list_ = [self.predict_y(each) for each in x_points_sample_list]
        #def py_given_x(x_points, y_point):
        def py_given_x(zip_list):
            x_points = zip_list[0]
            y_point = zip_list[1]
            #calculate p(y|x)
            #x_points with shape (sample_size, x_dim)
            #y_point wish shape (y_dim, )
            y_points_ = self.predict_y(x_points)
            return 1. / ((np.sqrt(2*np.pi)*sd_y)**self.y_dim) * np.exp(-(np.sum((y_point-y_points_)**2,axis=1))/(2.*sd_y**2))
        #def w_likelihood_ratio(x_point,x_points):
        def w_likelihood_ratio(zip_list):
            x_point = zip_list[0]
            x_points = zip_list[1]
            #calculate w=px/py
            #x_point with shape (x_dim, )
            #x_points with shape (sample_size,x_dim)
            qx =np.prod(t.pdf(x_point-x_points,self.df,loc=0,scale=self.scale),axis=1)
            px = 1. / (np.sqrt(2*np.pi)**self.x_dim) * np.exp(-(np.sum((x_points)**2,axis=1))/2.)
            return px / qx
        #calculate p(y|x) with multi-process
        pool = ThreadPool()
        #py_given_x_list = map(py_given_x, x_points_sample_list, y_points)
        py_given_x_list = pool.map(py_given_x, zip(x_points_sample_list, y_points))
        pool.close()
        pool.join()
        #calculate w=p(x)/q(x) with multi-process
        pool = ThreadPool()
        #w_likelihood_ratio_list = map(w_likelihood_ratio, x_points_, x_points_sample_list)
        w_likelihood_ratio_list = pool.map(w_likelihood_ratio, zip(x_points_, x_points_sample_list))
        pool.close()
        pool.join()
        #calculate p(y)=int(p(y|x)*p(x)dx)=int(p(y|x)*w(x)q(x)dx)=E(p(y|x)*w(x)) where x~q(x)
        py_list = map(lambda x, y: x*y,py_given_x_list,w_likelihood_ratio_list)

        py_est = np.array([np.mean(item) for item in py_list])
        print('Total time: %.1f s'%(time.time()-t0))
        if save:
            np.savez('%s/py_est_at_epoch%d.npz'%(self.save_dir,epoch),py_est,self.y_sampler.mean,self.y_sampler.sd, y_points)
        return py_est


    def evaluate(self,epoch):
        data_x, _ = self.x_sampler.load_all()
        data_y, _ = self.y_sampler.load_all()
        data_x_ = self.predict_x(data_y)
        data_y_ = self.predict_y(data_x)
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, epoch),data_x,data_y,data_x_,data_y_)
        self.plot_density_2D([data_x,data_y,data_x_,data_y_], '%s/figs'%self.save_dir, epoch)

    def plot_density_2D(self,data,save_dir,epoch,dim1=0,dim2=1):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        x, y, x_, y_ = data
        plt.figure() 
        plt.subplot(2,2,1)
        plt.hist2d(x[:,dim1],x[:,dim2],bins=200)
        plt.title('2D density of x')  
        plt.subplot(2,2,2)
        plt.hist2d(x_[:,dim1],x_[:,dim2],bins=200)
        plt.title('2D density of x*')  
        plt.subplot(2,2,3)
        plt.hist2d(y[:,dim1],y[:,dim2],bins=200)
        plt.title('2D density of y')  
        plt.subplot(2,2,4)
        plt.hist2d(y_[:,dim1],y_[:,dim2],bins=200)
        plt.title('2D density of y*')  
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('%s/2D_density_%d_%d_%s.png'%(save_dir,dim1,dim2,epoch))
        plt.close()




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
    parser.add_argument('--model', type=str, default='model_cycle_cluster')
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
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
    alpha = args.alpha
    beta = args.beta
    sd_y = args.sd_y
    df = args.df
    scale = args.scale
    timestamp = args.timestamp

    #g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=6,nb_units=256)
    #h_net = model.Generator(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=6,nb_units=256)
    g_net = model.Generator_resnet(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=256)
    h_net = model.Generator_resnet(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=10,nb_units=256)
    dx_net = model.Discriminator(input_dim = x_dim,name='dx_net',nb_layers=4,nb_units=256)
    dy_net = model.Discriminator(input_dim = y_dim,name='dy_net',nb_layers=4,nb_units=256)
    #xs = util.Y_sampler(N=10000, n_components=2,dim=y_dim,mean=3.0,sd=0.5)
    #ys = util.Y_sampler(N=10000, n_components=2,dim=x_dim,mean=0.0,sd=1)
    xs = util.Gaussian_sampler(N=10000,mean=np.zeros(x_dim),sd=1.0)
    #xs = util.X_sampler(N=10000, dim=x_dim, mean=3.0)
    #ys = util.Gaussian_sampler(N=10000,mean=np.zeros(y_dim),sd=1.0)
    ys = util.GMM_sampler(N=10000,n_components=5,dim=y_dim,sd=0.5)

    pool = util.DataPool()
    ################ gaussian mixture with 2 components############
    # mean = np.array([[0.25, 0.25],[0.75, 0.75]])
    # cov1 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov2 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov = np.array([cov1,cov2])
    # weights = [0.4,0.6]
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov,weights=weights)

    ################ gaussian mixture with 4 components############
    # mean = 0.75*np.array([[1, 1],[-1, 1],[1, -1],[-1, -1]])
    # sd = 0.05
    # cov = np.array([(sd**2)*np.eye(mean.shape[-1]) for item in range(len(mean))])
    # cov1 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov2 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov3 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov4 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov = np.array([cov1,cov2,cov3,cov4])
    # weights = [0.25,0.25,0.25,0.25]
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov,weights=weights)

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
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov)

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
        RTM.train()
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = 2999)
            #RTM.estimate_py_with_IS(2999)
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            t0=time.time()
            y_mean = ys.mean
            y_idx = np.random.choice(y_mean.shape[0], size=5000, replace=True)
            y_val = np.array([np.random.normal(y_mean[i],scale=0.5) for i in y_idx],dtype='float64')
            mle=[]
            for sd in np.linspace(0.1,1,19):
                print sd
                py_est = RTM.estimate_py_with_IS(y_val,1999,sd_y=sd)
                sys.exit()
                mle.append(np.sum(np.log(py_est+1e-100)))
            print mle
            plt.plot(np.linspace(0.1,1,19),mle)
            plt.xlabel('sd')
            plt.ylabel('log-likelihood')
            plt.savefig('model_select_at_dim%d.png'%(y_mean.shape[1]))
            best_sd = np.linspace(0.1,1,19)[mle.index(np.max(mle))]
            print('%.2f seconds'%(time.time()-t0))
            print('Best sd for RTM:%.2f'%best_sd)
            


    


