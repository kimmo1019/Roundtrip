#use cycle gan density estimation framework to do clustering by introducing discrete label
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
import copy
import metric
import util
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

tf.set_random_seed(0)

class ImagePool(object):
    def __init__(self, maxsize=30):
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

            idx = int(np.random.rand()*self.maxsize)
            tmp3 = copy.copy(self.images[idx])[2]
            self.images[idx][2] = image[2]
            return [tmp1, tmp2, tmp3]
        else:
            return image


class sysmGAN(object):
    def __init__(self, g_net, h_net, dx_net, dx_net_cat, dy_net, x_sampler, y_sampler, batch_size, nb_classes, alpha, beta):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dx_net_cat = dx_net_cat
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.beta = beta
        self.pool = ImagePool()
        self.use_L1=False
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
        self.x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot')
        self.x_combine = tf.concat([self.x,self.x_onehot],axis=1)

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x_combine)
        self.x__, self.x_onehot__, self.x_logits__ = self.h_net(self.y_)

        self.x_, self.x_onehot_, self.x_logits_ = self.h_net(self.y)#continuous + softmax + before_softmax
        self.x_combine_ = tf.concat([self.x_, self.x_onehot_],axis=1)

        self.y__ = self.g_net(self.x_combine_)

        self.dy_ = self.dy_net(self.y_, reuse=False)
        self.dy__ = self.dy_net(self.y__)

        self.dx_ = self.dx_net(self.x_, reuse=False)
        self.dx_onehot_ = self.dx_net_cat(self.x_onehot_, reuse=False)

        self.l1_loss_x = tf.reduce_mean(tf.abs(self.x - self.x__))
        self.l1_loss_y = tf.reduce_mean(tf.abs(self.y - self.y__))

        self.l2_loss_x = tf.reduce_mean((self.x - self.x__)**2)
        self.l2_loss_y = tf.reduce_mean((self.y - self.y__)**2)

        self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits__,labels=self.x_onehot))

        #-D(x)
        #self.g_loss_adv = -tf.reduce_mean(self.dy_)
        #self.h_loss_adv = -tf.reduce_mean(self.dx_)
        #(1-D(x))^2
        #self.g_loss_adv = tf.reduce_mean((tf.ones_like(self.dy_)  - self.dy_)**2)
        #self.h_loss_adv = tf.reduce_mean((tf.ones_like(self.dx_) - self.dx_)**2)
        #cross_entropy
        self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        self.h_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_)))
        if self.use_L1:
            self.g_loss = self.g_loss_adv + self.alpha*(self.CE_loss_x + self.l1_loss_x) + self.beta*self.l1_loss_y
            self.h_loss = self.h_loss_adv + self.alpha*(self.CE_loss_x + self.l1_loss_x) + self.beta*self.l1_loss_y
            self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.CE_loss_x + self.l1_loss_x) + self.beta*self.l1_loss_y
        else:
            self.g_loss = self.g_loss_adv + self.alpha*(self.CE_loss_x + self.l2_loss_x) + self.beta*self.l2_loss_y
            self.h_loss = self.h_loss_adv + self.alpha*(self.CE_loss_x + self.l2_loss_x) + self.beta*self.l2_loss_y
            self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.CE_loss_x + self.l2_loss_x) + self.beta*self.l2_loss_y
            #self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
            #self.g_h_loss = self.h_loss_adv + self.beta*self.l2_loss_y

        #self.fake_x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot')
        #self.fake_x = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x')
        #self.fake_x_combine = tf.concat([self.fake_x, self.fake_x_onehot],axis=1)
        
        #self.fake_y = tf.placeholder(tf.float32, [None, self.y_dim], name='fake_y')

        self.dx = self.dx_net(self.x)
        self.dx_onehot = self.dx_net_cat(self.x_onehot)
        self.dy = self.dy_net(self.y)

        #self.d_fake_x = self.dx_net(self.fake_x)
        #self.d_fake_y = self.dy_net(self.fake_y)


        #-D(x)
        #self.dx_loss = tf.reduce_mean(self.dx_) - tf.reduce_mean(self.dx)
        #self.dy_loss = tf.reduce_mean(self.dy_) - tf.reduce_mean(self.dy)
        #(1-D(x))^2
        # self.dx_loss = (tf.reduce_mean((tf.ones_like(self.dx) - self.dx)**2) \
        #         +tf.reduce_mean((tf.zeros_like(self.d_fake_x) - self.d_fake_x)**2))/2.0
        # self.dy_loss = (tf.reduce_mean((tf.ones_like(self.dy) - self.dy)**2) \
        #         +tf.reduce_mean((tf.zeros_like(self.d_fake_y) - self.d_fake_y)**2))/2.0
        self.dx_cont_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx, labels=tf.ones_like(self.dx))) \
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.zeros_like(self.dx_)))
        self.dx_cat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_onehot, labels=tf.ones_like(self.dx_onehot))) \
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_onehot_, labels=tf.zeros_like(self.dx_onehot_)))
        self.dx_loss = self.dx_cont_loss+self.dx_cat_loss

        self.gx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_))) \
                +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_onehot_, labels=tf.ones_like(self.dx_onehot_)))
        self.gy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        
        self.dy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy, labels=tf.ones_like(self.dy))) \
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.zeros_like(self.dy_)))
        self.d_loss = self.dx_loss + self.dy_loss

        self.clip_dx = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net.vars]
        self.clip_dy = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net.vars]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        # self.g_h_optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr) \
        #         .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        # self.d_optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr) \
        #         .minimize(self.d_loss, var_list=self.dx_net.vars+self.dx_net_cat.vars+self.dy_net.vars)
        #self.h_optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr) \
        #        .minimize(self.d_loss, var_list=self.h_net.vars)   
        # self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        # self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.d_loss, var_list=self.dx_net.vars+self.dx_net_cat.vars+self.dy_net.vars)

        self.recon_y = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.l2_loss_y, var_list=self.g_net.vars+self.h_net.vars)
        self.dis_x = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dx_loss, var_list=self.dx_net.vars+self.dx_net_cat.vars)
        self.gen_x = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.gx_loss, var_list=self.h_net.vars)

        self.recon_x = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.l2_loss_x+self.CE_loss_x, var_list=self.g_net.vars+self.h_net.vars)
        self.dis_y = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dy_loss, var_list=self.dy_net.vars)
        self.gen_y = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.gy_loss, var_list=self.g_net.vars)

        # self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        # self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
        #         .minimize(self.d_loss, var_list=self.dx_net.vars+self.dx_net_cat.vars+self.dy_net.vars)

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

        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, epochs=500):
        #data_x, label_x = self.x_sampler.load_all()
        data_y, label_y = self.y_sampler.load_all()
        #data_x = np.array(data_x,dtype='float32')
        data_y = np.array(data_y,dtype='float32')
        batch_size = self.batch_size
        counter=1
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for epoch in range(epochs):
            #np.random.shuffle(data_x)
            np.random.shuffle(data_y)
            lr = 1e-4 #if epoch < epochs/2 else 1e-4#2e-4*(epochs-epoch)/(epochs-epochs/2)
            batch_idxs = len(data_y) // batch_size
            #lr decay, add later
            for idx in range(batch_idxs):
                #bx = data_x[batch_size*idx:batch_size*(idx+1)]
                #bx, bx_onehot = self.x_sampler.get_batch(batch_size)
                bx, bx_onehot = self.x_sampler.train(batch_size)
                by = data_y[batch_size*idx:batch_size*(idx+1)]
                #update G and get generated fake data
                #fake_bx, fake_bx_onehot, fake_by, _ = self.sess.run([self.x_,self.x_onehot_, self.y_,self.g_h_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #_ = self.sess.run(self.g_h_optim, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #_ = self.sess.run(self.d_optim, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                
                _ = self.sess.run(self.recon_y, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _ = self.sess.run(self.dis_x, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _ = self.sess.run(self.gen_x, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _ = self.sess.run(self.recon_x, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _ = self.sess.run(self.dis_y, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                _ = self.sess.run(self.gen_y, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #_ = self.sess.run(self.dis_y, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #_ = self.sess.run(self.gen_y, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #random choose one batch from the previous 50 batches,flash back
                #[fake_bx,fake_bx_onehot,fake_by] = self.pool([fake_bx,fake_bx_onehot,fake_by])

                #update D
                #self.sess.run(self.d_optim, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #update H
                #self.sess.run(self.h_optim, feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
                #self.sess.run(self.d_optim, feed_dict={self.x: bx, self.x_onehot:bx_onehot,self.y: by, self.fake_x: fake_bx, self.fake_x_onehot:fake_bx_onehot ,self.fake_y: fake_by,self.lr:lr})
                #self.sess.run([self.clip_dx, self.clip_dy])
                #quick test on a random batch data
                if counter % 100 == 0:
                    #bx, bx_onehot = self.x_sampler.get_batch(batch_size)
                    bx, bx_onehot = self.x_sampler.train(batch_size)
                    by = self.y_sampler.train(batch_size)

                    g_loss_adv, h_loss_adv, CE_loss_x, l2_loss_x, l2_loss_y, g_loss, \
                        h_loss, g_h_loss, fake_bx, fake_by = self.sess.run(
                        [self.g_loss_adv, self.h_loss_adv, self.CE_loss_x, self.l2_loss_x, self.l2_loss_y, \
                        self.g_loss, self.h_loss, self.g_h_loss, self.x_, self.y_],
                        feed_dict={self.x: bx, self.x_onehot:bx_onehot, self.y: by}
                    )
                    dx_cont_loss, dx_cat_loss, dx_loss, dy_loss, d_loss = self.sess.run([self.dx_cont_loss,self.dx_cat_loss,self.dx_loss, self.dy_loss, self.d_loss], \
                        feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by})

                    print('Iter [%8d] Time [%5.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss_x [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dx_cont_loss [%.4f] dx_cat_loss [%.4f] dy_loss [%.4f] d_loss [%.4f]' %
                        (counter, time.time() - start_time, g_loss_adv, h_loss_adv, CE_loss_x, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dx_cont_loss, dx_cat_loss ,dy_loss, d_loss))                    
                counter+=1
                
                # summary = self.sess.run(self.merged_summary,feed_dict={self.x:bx,self.y:by})
                # self.summary_writer.add_summary(summary, counter)
            if (epoch+1) % 20 == 0:
                self.evaluate(timestamp,epoch)

        #self.save(timestamp)

    def evaluate(self,timestamp,epoch):
        #use all data
        #data_x, data_x_onehot, _ = self.x_sampler.load_all()
        data_y, label_y = self.y_sampler.load_all()
        #assert data_x.shape[0] == data_y.shape[0]
        N = data_y.shape[0]
        data_x_ = np.zeros(shape=(N, self.x_dim))
        data_x_onehot_ =  np.zeros(shape=(N, self.nb_classes))
        #data_y_ = np.zeros(shape=(N, self.y_dim))

        for b in range(int(np.ceil(N*1.0 / self.batch_size))):

            if (b+1)*self.batch_size > N:
               ind = np.arange(b*self.batch_size, N)
            else:
               ind = np.arange(b*self.batch_size, (b+1)*self.batch_size)
            #batch_x = data_x[ind, :]
            #batch_x_onehot = data_x_onehot[ind, :]
            batch_y = data_y[ind, :]
            #batch_x_, batch_x_onehot_, batch_y_ = self.sess.run([self.x_,self.x_onehot_ ,self.y_], feed_dict={self.x:batch_x,self.x_onehot:batch_x_onehot, self.y:batch_y})
            batch_x_, batch_x_onehot_ = self.sess.run([self.x_,self.x_onehot_ ], feed_dict={self.y:batch_y})
            data_x_[ind, :] = batch_x_
            data_x_onehot_[ind, :] = batch_x_onehot_
            #data_y_[ind, :] = batch_y_
        save_dir = 'data/density_est/{}_{}_{}_{}_{}'.format(self.timestamp,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #np.savez('{}/data_at_{}.npz'.format(save_dir, epoch+1),data_x,data_y,data_x_,data_y_)
        np.savez('{}/data_at_{}.npz'.format(save_dir, epoch+1),data_x_,data_x_onehot_)
        #self.plot_density_2D([data_x,data_y,data_x_,data_y_], '%s/figs'%save_dir, epoch)
        label_infer = np.argmax(data_x_onehot_, axis=1)
        purity = metric.compute_purity(label_infer, label_y)
        nmi = normalized_mutual_info_score(label_y, label_infer)
        ari = adjusted_rand_score(label_y, label_infer)
        print('DeepSC: Purity = {}, NMI = {}, ARI = {}'.format(purity,nmi,ari))
        f = open('%s/log.txt'%save_dir,'a+')
        f.write('%.4f\t%.4f\t%.4f\t%d\n'%(purity,nmi,ari,epoch))
        f.close()
        #k-means
        if (epoch+1) % 1000 ==0:
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y)
            nmi = normalized_mutual_info_score(label_y, label_kmeans)
            ari = adjusted_rand_score(label_y, label_kmeans)
            print('K-means: Purity = {}, NMI = {}, ARI = {}'.format(purity,nmi,ari))
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
        plt.title('2D density of x')  
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('%s/2D_density_%d_%d_%s.png'%(save_dir,dim1,dim2,epoch))


    def save(self, timestamp):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_{}_{}'.format(self.data, timestamp, self.x_dim,
                                                                             self.y_dim, self.alpha,
                                                                             self.beta)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))

    def load(self, pre_trained = False, timestamp = ''):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_{}_{}}'.format(self.data, self.x_dim,
                                                                            self.y_dim, self.alpha, self.beta)
        else:
            if timestamp == '':
                print('Best Timestamp not provided. Abort !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_{}'.format(self.data, self.x_dim,
                                                                            self.y_dim, self.alpha, self.beta)


        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        print('Restored model weights.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='simutation_data')
    parser.add_argument('--model', type=str, default='model_cycle_cluster')
    parser.add_argument('--K', type=int, default=11)
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
    nb_classes = args.K
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    alpha = args.alpha
    beta = args.beta
    timestamp = args.timestamp
    # z_dim = dim_gen + num_classes * n_cat
    # d_net = model.Discriminator(x_dim = x_dim)
    # g_net = model.Generator(z_dim=z_dim,x_dim = x_dim)
    # enc_net = model.Encoder(z_dim=z_dim, dim_gen = dim_gen,x_dim = x_dim)
    g_net = model.Generator(x_dim+nb_classes, y_dim, 'g_net')
    #g_net = model.Generator(x_dim, y_dim, 'g_net')
    #h_net = model.Generator(y_dim, x_dim+nb_classes, name='h_net')
    h_net = model.Encoder(y_dim, x_dim+nb_classes, x_dim, 'h_net')
    dx_net = model.Discriminator(x_dim, 'dx_net')
    dx_net_cat = model.Discriminator(nb_classes, 'dx_net_cat')
    dy_net = model.Discriminator(y_dim, 'dy_net')

    xs = util.Mixture_sampler_v2(nb_classes=nb_classes,N=10000,dim=x_dim)
    #ys = util.DataSampler() #scRNA-seq data
    #xs = util.Y_sampler(N=10000, n_components=2,dim=y_dim,mean=3.0,sd=0.5)
    #ys = util.Y_sampler(N=10000, n_components=2,dim=x_dim,mean=0.0,sd=1)
    #xs = util.Gaussian_sampler(N=10000,mean=np.zeros(x_dim),sd=1.0)
    #ys = util.UniformDataSampler(N=10000,n_components=10,dim=y_dim,sd=1)
    ys = util.GMM_sampler(N=10000,n_components=10,dim=y_dim,sd=1)
    #xs = util.X_sampler(N=10000, dim=x_dim, mean=3.0)
    #ys = util.Gaussian_sampler(N=10000,mean=np.zeros(y_dim),sd=1.0)
    # mean = np.array([[0.25, 0.25],[0.75, 0.75]])
    # cov1 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov2 = np.array([[0.05**2, 0],[0, 0.05**2]])
    # cov = np.array([cov1,cov2])
    # weights = [0.4,0.6]
    # ys = util.GMM_sampler(N=10000,mean=mean,cov=cov,weights=weights)


    cl_gan = sysmGAN(g_net, h_net, dx_net, dx_net_cat, dy_net, xs, ys, batch_size, nb_classes, alpha, beta)

    if args.train == 'True':
        cl_gan.train()
    else:

        print('Attempting to Restore Model ...')
        if timestamp == '':
            cl_gan.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            cl_gan.load(pre_trained=False, timestamp = timestamp)


