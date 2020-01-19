import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
import copy
import math
import metric
import util
from functools import reduce

tf.set_random_seed(0)

class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


class sysmGAN(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, batch_size, alpha, beta):
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
        self.pool = ImagePool()
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
        save_dir = 'data/density_est/density_est_{}_{}_{}_{}_{}'.format(self.timestamp,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, epochs=2000):
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
            lr = 1e-4 #if epoch < epochs/2 else 2e-4*(epochs-epoch)/(epochs-epochs/2)
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

                    g_loss_adv, h_loss_adv, l1_loss_x, l1_loss_y, g_loss, \
                        h_loss, g_h_loss, fake_bx, fake_by = self.sess.run(
                        [self.g_loss_adv, self.h_loss_adv, self.l1_loss_x, self.l1_loss_y, \
                        self.g_loss, self.h_loss, self.g_h_loss, self.x_, self.y_],
                        feed_dict={self.x: bx, self.y: by}
                    )
                    dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                        feed_dict={self.x: bx, self.y: by, self.fake_x: fake_bx, self.fake_y: fake_by})

                    print('Iter [%8d] Time [%5.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] l1_loss_x [%.4f] \
                        l1_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f]' %
                        (counter, time.time() - start_time, g_loss_adv, h_loss_adv, l1_loss_x, l1_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
                counter+=1
                
                # summary = self.sess.run(self.merged_summary,feed_dict={self.x:bx,self.y:by})
                # self.summary_writer.add_summary(summary, counter)
            if (epoch+1) % 100 == 0:
                self.evaluate(epoch)
                #self.density_est(epoch)
                self.save(epoch)
            if (epoch+1) % 500 == 0:
                self.estimate_fy_with_IS_v3(epoch)#density estimation with importance sampling

    #predict with y_=G(x)
    def predict_y(self, x, bs=512):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.x_dim)) 
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
    def predict_x(self,y,bs=512):
        assert y.shape[-1] == self.y_dim
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):

            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_ = self.sess.run(self.x_, feed_dict={self.y:batch_y})
            x_pred[ind, :] = batch_x_
        return x_pred

    
    def estimate_fy_with_IS(self,epoch,sd_q=1,n=200,interval_len=10,sample_size=5000):
        #importace sampling with normal distribution (2 dimension)
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        def f(x_batch,sd_y=0.1):
            return 1. / ((np.sqrt(2 * np.pi)*sd_y)**self.y_dim) * np.exp(-(np.sum(x_batch**2,axis=1))/(2*sd_y**2))
        def w(x,x_y,sd_q=1):
            mu1 = np.sum((x-x_y)**2,axis=1) #length=sample_size
            #mu1 = np.sum(x**2,axis=1)
            mu2 = np.sum(x**2,axis=1) 
            return (sd_q**self.x_dim)*np.exp(mu1/(2*sd_q**2)-mu2/2)
        grid_axis1 = np.linspace(-interval_len/2,interval_len/2,n)
        grid_axis2 = np.linspace(-interval_len/2,interval_len/2,n)
        v1,v2 = np.meshgrid(grid_axis1,grid_axis2)
        y_points = np.vstack((v1.ravel(),v2.ravel())).T#shape (N,2)
        x_points_pred = self.predict_x(y_points)
        #x_points_pred = np.zeros((n**2,self.x_dim))
        x_samples_list = [np.random.normal(each, sd_q,(sample_size,self.x_dim)) for each in x_points_pred]
        #x_samples_list = [np.random.normal(0, 1,(sample_size,self.x_dim)) for each in x_points_pred]
        g_x_samples_list = [self.predict_y(each) for each in x_samples_list]
        #np.save('pre.npy',g_x_samples_list[0])
        #g_x_samples_list = [np.zeros((sample_size,self.y_dim)) for each in x_samples_list]
        #y-G(x)
        mu_samples_list = map(lambda x: x[0]-x[1], zip(y_points, g_x_samples_list))
        f_return_list = map(f,mu_samples_list)
        w_return_list = map(w,x_samples_list,x_points_pred)
        fy_list = map(lambda x, y: x*y,f_return_list,w_return_list)
        fy_est = np.array([np.mean(item) for item in fy_list])
        #fy_est = np.array([np.mean(item) for item in f_return_list])
        fy_est = fy_est.reshape((n,n))
        #(time.time()-t)
        plt.figure()
        plt.pcolormesh(v1,v2,fy_est,cmap='coolwarm')
        plt.colorbar()
        plt.savefig('density_y_at_epoch%d_%.2f.png'%(epoch,time.time()-t))
        plt.close()

    def estimate_fy_with_IS_v2(self,epoch,degree=1,n=200,interval_len=10,sample_size=5000):#using student t distribution
        #importace sampling with t-distribution (2 dimension)
        sd_q = 1
        from scipy.stats import t
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        def f(x_batch,sd_y=0.1):
            return 1. / ((np.sqrt(2 * np.pi)*sd_y)**self.y_dim) * np.exp(-(np.sum(x_batch**2,axis=1))/(2*sd_y**2))
        def w(x,x_y,degree=1):
            #q_x = 1. / ((np.sqrt(2 * np.pi)*sd_q)**self.x_dim) * np.exp(-np.sum((x-x_y)**2,axis=1)/(2*sd_q**2))
            t_pdf = t.pdf(x-x_y,degree) #shape sample_size * x_dim
            q_x = np.prod(t_pdf,axis=1)
            p_x = 1. / ((np.sqrt(2 * np.pi))**self.x_dim) * np.exp(-np.sum(x**2,axis=1)/2) #length=sample_size
            return p_x/q_x
        grid_axis1 = np.linspace(-interval_len/2,interval_len/2,n)
        grid_axis2 = np.linspace(-interval_len/2,interval_len/2,n)
        v1,v2 = np.meshgrid(grid_axis1,grid_axis2)
        y_points = np.vstack((v1.ravel(),v2.ravel())).T#shape (N,2)
        x_points_pred = self.predict_x(y_points)
        #x_points_pred = np.zeros((n**2,self.x_dim))
        
        #x_samples_list = [np.random.normal(each, sd_q,(sample_size,self.x_dim)) for each in x_points_pred]
        x_samples_list = [np.hstack([t.rvs(degree, loc=item, scale=1, size=(sample_size,1)) for item in each]) for each in x_points_pred]
        g_x_samples_list = [self.predict_y(each) for each in x_samples_list]
        #np.save('pre.npy',g_x_samples_list[0])
        #g_x_samples_list = [np.zeros((sample_size,self.y_dim)) for each in x_samples_list]
        #y-G(x)
        mu_samples_list = map(lambda x: x[0]-x[1], zip(y_points, g_x_samples_list))
        f_return_list = map(f,mu_samples_list)
        w_return_list = map(w,x_samples_list,x_points_pred)
        fy_list = map(lambda x, y: x*y,f_return_list,w_return_list)
        fy_est = np.array([np.mean(item) for item in fy_list])
        #fy_est = np.array([np.mean(item) for item in f_return_list])
        fy_est = fy_est.reshape((n,n))
        #(time.time()-t)
        plt.figure()
        plt.pcolormesh(v1,v2,fy_est,cmap='coolwarm')
        plt.colorbar()
        plt.savefig('density_y_at_epoch%d_%s.png'%(epoch,self.timestamp))
        plt.close()


    def estimate_fy_with_IS_v3(self,epoch,degree=1,n=200,interval_len=2,sample_size=5000):#using student t distribution
        #importace sampling with t-distribution (any dimension)
        t0=time.time()
        sd_q = 1
        from scipy.stats import t
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        def f(x_batch,sd_y=0.1):
            return 1. / ((np.sqrt(2 * np.pi)*sd_y)**self.y_dim) * np.exp(-(np.sum(x_batch**2,axis=1))/(2*sd_y**2))
        def w(x,x_y,degree=1):
            #q_x = 1. / ((np.sqrt(2 * np.pi)*sd_q)**self.x_dim) * np.exp(-np.sum((x-x_y)**2,axis=1)/(2*sd_q**2))
            t_pdf = t.pdf(x-x_y,degree) #shape sample_size * x_dim
            q_x = np.prod(t_pdf,axis=1)
            p_x = 1. / ((np.sqrt(2 * np.pi))**self.x_dim) * np.exp(-np.sum(x**2,axis=1)/2) #length=sample_size
            return p_x/q_x

        linspace_list = [np.linspace(-interval_len/2,interval_len/2,n) for _ in range(self.y_dim)]
        mesh_grids_list = np.meshgrid(*linspace_list)
        y_points = np.vstack([item.ravel() for item in mesh_grids_list]).T
        x_points_pred = self.predict_x(y_points)
        
        x_samples_list = [np.hstack([t.rvs(degree, loc=item, scale=1, size=(sample_size,1)) for item in each]) for each in x_points_pred]
        g_x_samples_list = [self.predict_y(each) for each in x_samples_list]
        #y-G(x)
        mu_samples_list = map(lambda x: x[0]-x[1], zip(y_points, g_x_samples_list))
        f_return_list = map(f,mu_samples_list)
        w_return_list = map(w,x_samples_list,x_points_pred)
        fy_list = map(lambda x, y: x*y,f_return_list,w_return_list)
        fy_est = np.array([np.mean(item) for item in fy_list])
        fy_est = fy_est.reshape(tuple([n]*self.y_dim))
        print time.time()-t0
        np.save('%s/py_est_at_epoch%d_%.2f.npy'%(self.save_dir,epoch,time.time()-t0),fy_est)
        # plt.figure()
        # plt.pcolormesh(v1,v2,fy_est,cmap='coolwarm')
        # plt.colorbar()
        # plt.savefig('density_y_at_epoch%d_%s.png'%(epoch,self.timestamp))
        # plt.close()

    def density_est(self,epoch,n=500,bound=5):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        grid_axis1 = np.linspace(-bound,bound,n)#axis-dim1
        grid_axis2 = np.linspace(-bound,bound,n)#axis-dim2
        N = n**2
        xv1,xv2 = np.meshgrid(grid_axis1,grid_axis2)
        grid_x = np.vstack((xv1.ravel(),xv2.ravel())).T
        grid_y_ = self.predict_y(grid_x)
        def calculate_product(point_y,xv1,xv2,grid_y_,sd=0.05):
            f1_mat = 1. / (np.sqrt((2 * np.pi)**self.x_dim)) * np.exp(-(xv1**2+xv2**2) / 2)#with shape n*n
            l2_norm = np.linalg.norm(point_y-grid_y_,ord=2,axis=1)
            f2_list = 1. / ((np.sqrt(2 * np.pi)*sd)**self.y_dim) * np.exp(-(l2_norm**2) / (2*sd**2))
            f2_mat = f2_list.reshape((len(f1_mat),-1))
            return f1_mat,f2_mat
        y_list = 0.75*np.array([[-1,1],[-1,0],[-1,-1],[0,1],[0,0],[0,-1],[1,1],[1,0],[1,-1]],dtype='float')
        for i in range(0,len(y_list),3):
            plt.figure(figsize=(12, 10),dpi=100)
            for j in range(i,i+3):
                f1_mat,f2_mat = calculate_product(y_list[j],xv1,xv2,grid_y_)
                plt.subplot(3,3,3*(j%3)+1)
                plt.pcolormesh(xv1,xv2,f1_mat,cmap='coolwarm')
                plt.title('y=(%.2f,%.2f)'%(y_list[j][0],y_list[j][1]))
                plt.colorbar()
                plt.subplot(3,3,3*(j%3)+2)
                plt.pcolormesh(xv1,xv2,f2_mat,cmap='coolwarm')
                plt.title('y=(%.2f,%.2f)'%(y_list[j][0],y_list[j][1]))
                plt.colorbar()
                plt.subplot(3,3,3*(j%3)+3)
                plt.pcolormesh(xv1,xv2,f1_mat*f2_mat,cmap='coolwarm')
                plt.title('y=(%.2f,%.2f)'%(y_list[j][0],y_list[j][1]))
                plt.colorbar()
            plt.savefig('%s/map_y1=%d_at_epoch%d'%(self.save_dir,4*(i/3-1),epoch))
            plt.close()

    def evaluate(self,epoch):
        #use all data
        data_x, _ = self.x_sampler.load_all()
        data_y, _ = self.y_sampler.load_all()
        #assert data_x.shape[0] == data_y.shape[0]
        data_x_ = self.predict_x(data_y)
        data_y_ = self.predict_y(data_x)
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, epoch),data_x,data_y,data_x_,data_y_)
        self.plot_density_2D([data_x,data_y,data_x_,data_y_], '%s/figs'%self.save_dir, epoch)
        #sys.exit()
        #calculate KL-divergency of data_x_ and data_x, data_y_ and data_y

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
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
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
    timestamp = args.timestamp

    g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net')
    h_net = model.Generator(input_dim=y_dim,output_dim = x_dim,name='h_net')
    dx_net = model.Discriminator(input_dim = x_dim,name='dx_net')
    dy_net = model.Discriminator(input_dim = y_dim,name='dy_net')
    #xs = util.Y_sampler(N=10000, n_components=2,dim=y_dim,mean=3.0,sd=0.5)
    #ys = util.Y_sampler(N=10000, n_components=2,dim=x_dim,mean=0.0,sd=1)
    xs = util.Gaussian_sampler(N=10000,mean=np.zeros(x_dim),sd=1.0)
    #xs = util.X_sampler(N=10000, dim=x_dim, mean=3.0)
    #ys = util.Gaussian_sampler(N=10000,mean=np.zeros(y_dim),sd=1.0)

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
    linspace_list = 0.75*np.array([np.linspace(-1.,1.,2) for _ in range(y_dim)])
    mesh_grids_list = np.meshgrid(*linspace_list)
    mean = np.vstack([item.ravel() for item in mesh_grids_list]).T
    sd = 0.05
    cov = np.array([(sd**2)*np.eye(mean.shape[-1]) for item in range(len(mean))])
    ys = util.GMM_sampler(N=10000,mean=mean,cov=cov)
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

    



    cl_gan = sysmGAN(g_net, h_net, dx_net, dy_net, xs, ys, batch_size, alpha, beta)

    if args.train == 'True':
        cl_gan.train()
    else:

        print('Attempting to Restore Model ...')
        if timestamp == '':
            cl_gan.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            cl_gan.load(pre_trained=False, timestamp = timestamp, epoch = 999)
            cl_gan.estimate_fy_with_IS_v2(999)
            sys.exit()
            grid_axis1 = np.linspace(-5,5,200)#axis-dim1
            grid_axis2 = np.linspace(-5,5,200)#axis-dim2
            xv1,xv2 = np.meshgrid(grid_axis1,grid_axis2)
            data = np.vstack((xv1.ravel(),xv2.ravel())).T
            #data_x = xs.get_batch(300000)
            data_y_ = cl_gan.predict_y(data)
            data_x_ = cl_gan.predict_x(data)
            np.savez('data.npz',data,data_x_)
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt 
            plt.figure()
            plt.hist2d(data_x_[:,0],data_x_[:,1],bins=8000)
            plt.colorbar()
            plt.xlim(-3,3)
            plt.ylim(-3,3)
            plt.savefig('test_x2.png')
            plt.close()
            #cl_gan.estimate_fy_with_IS(999)


    


