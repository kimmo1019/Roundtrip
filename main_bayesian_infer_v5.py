from __future__ import division
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import math
import metric
import util
import random

##########################################################################
'''
Instructions: Roundtrip model for Bayesian inference
It contains three steps training at each processing data block.
Step1 - train a standart Roundtrip model RTM(x1,y1)
Step2 - pretrain a GAN model GAN(x2,y2)
Step3 - train the extended GAN model GAN((x1,x2),(y1,y2)) where relationship
        between x1~y1 is modeled by RTM(x1,y1) and fixed

    x1,x2 - two independent variables from known distributions (e.g., Gaussian)
    y1,y2 - observation data and model parameters from Bayesian model
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)   - generator network for mapping x space to y space, note that it
             is a partially connected network which has two outputs e.g, 
             G((x1,x2))=(y1,y2) which equals to f1(x1)=y1 and f2((x1,x2))=y2.
             The trainable params in G will be divided into three groups (w1,
             w2,w3) x1~y1 is parametrized by w1, (x1,x2)~y2 is parametrized 
             by w2 and w3, note that w3 denotes cross connections
    H(.)   - generator network for mapping y space to x space, the same as G
             H is only used in step 1
    Dx1(.) - discriminator network in x space for step 1
    Dy1(.) - discriminator network in y space for step 1
    Dy2(.) - discriminator network in y space for step 2
    Dy(.)  - discriminator network in y space for step 3
'''
##########################################################################

class RoundtripModel(object):
    def __init__(self, g_net, h_net, dx_net1, dy_net1,dy_net2, dy_net, x_sampler1, x_sampler2, y_sampler, pool, batch_size, alpha, beta, epochs):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net1 = dx_net1
        self.dy_net1 = dy_net1
        self.dy_net2 = dy_net2
        self.dy_net = dy_net
        self.x_sampler1 = x_sampler1
        self.x_sampler2 = x_sampler2
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.pool = pool
        self.epochs = epochs
        self.x_dim1 = self.g_net.input_dim1
        self.x_dim2 = self.g_net.input_dim2
        self.y_dim1 = self.h_net.input_dim1
        self.y_dim2 = self.h_net.input_dim2

        tf.reset_default_graph()

        self.x1 = tf.placeholder(tf.float32, [None, self.x_dim1], name='x1')
        self.x2 = tf.placeholder(tf.float32, [None, self.x_dim2], name='x2')
        self.y1 = tf.placeholder(tf.float32, [None, self.y_dim1], name='y1')
        self.y2 = tf.placeholder(tf.float32, [None, self.y_dim2], name='y2')
        self.x = tf.concat([self.x1,self.x2],axis=1)
        self.y = tf.concat([self.y1,self.y2],axis=1)

        self.y1_, self.y2_ = self.g_net(self.x,reuse=False)
        self.x1_, self.x2_ = self.h_net(self.y,reuse=False)

        self.x_ = tf.concat([self.x1_,self.x2_],axis=1)
        self.y_ = tf.concat([self.y1_,self.y2_],axis=1)

        self.x1__, self.x2__ = self.h_net(self.y_)
        self.y1__, self.y2__ = self.g_net(self.x_)

        self.x__ = tf.concat([self.x1__,self.x2__],axis=1)
        self.y__ = tf.concat([self.y1__,self.y2__],axis=1)

        self.dx1_ = self.dx_net1(self.x1_, reuse=False)
        self.dy1_ = self.dy_net1(self.y1_, reuse=False)

        self.dy2_ = self.dy_net2(self.y2_, reuse=False)
        self.dy_ = self.dy_net(self.y_, reuse=False)
        

        self.l2_loss_x1 = tf.reduce_mean((self.x1 - self.x1__)**2)
        self.l2_loss_y1 = tf.reduce_mean((self.y1 - self.y1__)**2)  

        #(1-D(x))^2
        self.g_loss_adv1 = tf.reduce_mean((0.9*tf.ones_like(self.dy1_)  - self.dy1_)**2)
        self.h_loss_adv1 = tf.reduce_mean((0.9*tf.ones_like(self.dx1_) - self.dx1_)**2)
        
        self.g_loss_adv2 = tf.reduce_mean((0.9*tf.ones_like(self.dy2_)  - self.dy2_)**2)
        self.g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(self.dy_)  - self.dy_)**2)

        #cross_entropy
        #self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        #self.h_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_)))
        self.g_loss1 = self.g_loss_adv1 + self.alpha*self.l2_loss_x1 + self.beta*self.l2_loss_y1
        self.h_loss1 = self.h_loss_adv1 + self.alpha*self.l2_loss_x1 + self.beta*self.l2_loss_y1

        self.g_h_loss1 = self.g_loss_adv1 + self.h_loss_adv1 + self.alpha*self.l2_loss_x1 + self.beta*self.l2_loss_y1


        self.fake_x1 = tf.placeholder(tf.float32, [None, self.x_dim1], name='fake_x1')
        self.fake_x2 = tf.placeholder(tf.float32, [None, self.x_dim2], name='fake_x2')
        self.fake_y1 = tf.placeholder(tf.float32, [None, self.y_dim1], name='fake_y1')
        self.fake_y2 = tf.placeholder(tf.float32, [None, self.y_dim2], name='fake_y2')
        self.fake_x = tf.concat([self.fake_x1,self.fake_x2],axis=1)
        self.fake_y = tf.concat([self.fake_y1,self.fake_y2],axis=1)

        self.dx1 = self.dx_net1(self.x1)
        self.dy1 = self.dy_net1(self.y1)        
        
        self.dy2 = self.dy_net2(self.y2)
        self.dy = self.dy_net(self.y)

        self.dy_loss2 = (tf.reduce_mean((0.9*tf.ones_like(self.dy2) - self.dy2)**2) \
               +tf.reduce_mean((0.1*tf.zeros_like(self.dy2_) - self.dy2_)**2))/2.0
        self.dy_loss = (tf.reduce_mean((0.9*tf.ones_like(self.dy) - self.dy)**2) \
               +tf.reduce_mean((0.1*tf.zeros_like(self.dy_) - self.dy_)**2))/2.0
        
        self.d_fake_x1 = self.dx_net1(self.fake_x1)
        self.d_fake_y1 = self.dy_net1(self.fake_y1)
        
        #-D(x)
        #self.dx_loss = tf.reduce_mean(self.dx_) - tf.reduce_mean(self.dx)
        #self.dy_loss = tf.reduce_mean(self.dy_) - tf.reduce_mean(self.dy)
        #(1-D(x))^2
        self.dx_loss1 = (tf.reduce_mean((0.9*tf.ones_like(self.dx1) - self.dx1)**2) \
                +tf.reduce_mean((0.1*tf.zeros_like(self.d_fake_x1) - self.d_fake_x1)**2))/2.0
        self.dy_loss1 = (tf.reduce_mean((0.9*tf.ones_like(self.dy1) - self.dy1)**2) \
                +tf.reduce_mean((0.1*tf.zeros_like(self.d_fake_y1) - self.d_fake_y1)**2))/2.0
 
        self.d_loss1 = self.dx_loss1 + self.dy_loss1

        #self.clip_dx = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dx_net.vars]
        #self.clip_dy = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.dy_net.vars]
        
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        #stage1 RTM(x1,y1)
        self.g_h_optim1 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss1, var_list=self.g_net.vars[0]+self.h_net.vars[0])
        self.d_optim1 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss1, var_list=self.dx_net1.vars+self.dy_net1.vars)
        
        #stage2 pretrain GAN(x2,y2)
        self.g_optim2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss_adv2, var_list=self.g_net.vars[1])
        self.d_optim2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dy_loss2, var_list=self.dy_net2.vars)

        #stage3 train GAN((x1,x2),(y1,y2))
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss_adv, var_list=self.g_net.vars[1]+self.g_net.vars[2])
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dy_loss, var_list=self.dy_net.vars)


        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.g_loss_adv_summary = tf.summary.scalar('g_loss_adv1',self.g_loss_adv1)
        self.h_loss_adv_summary = tf.summary.scalar('h_loss_adv1',self.h_loss_adv1)
        self.l2_loss_x_summary = tf.summary.scalar('l2_loss_x1',self.l2_loss_x1)
        self.l2_loss_y_summary = tf.summary.scalar('l2_loss_y1',self.l2_loss_y1)
        self.dx_loss_summary = tf.summary.scalar('dx_loss1',self.dx_loss1)
        self.dy_loss_summary = tf.summary.scalar('dy_loss1',self.dy_loss1)
        self.g_merged_summary = tf.summary.merge([self.g_loss_adv_summary, self.h_loss_adv_summary,\
            self.l2_loss_x_summary,self.l2_loss_y_summary])
        self.d_merged_summary = tf.summary.merge([self.dx_loss_summary,self.dy_loss_summary])
        #graph path for tensorboard visualization
        self.graph_dir = 'graph/bayes_infer_{}_x_dim={}_{}_y_dim={}_{}_alpha={}_beta={}'.format(self.timestamp,self.x_dim1, self.x_dim2, self.y_dim1, self.y_dim2, self.alpha, self.beta)
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/bayes_infer/bayes_infer_{}_x_dim={}_{}_y_dim={}_{}_alpha={}_beta={}'.format(self.timestamp,self.x_dim1, self.x_dim2, self.y_dim1, self.y_dim2, self.alpha, self.beta)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=100)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)
    
    def bayesian_iteration(self, sample_size=500000, n_iters=10):
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        
        #self.y_true, self.theta_true = self.y_sampler.generate_data(sample_size=1,time_step=100)
        #y_prior, theta_prior = self.y_sampler.generate_data(sample_size)
        y_prior, theta_prior, _ = self.y_sampler.generate_data3(sample_size, 0, self.y_dim1)
        self.y_true = y_prior[0:1,:]
        np.savez('%s/stage0_init.npz'%self.save_dir,y_prior,theta_prior)
        data_y1 = y_prior
        data_y2 = theta_prior
        theta_posterior = self.train(data_y1,data_y2,0,epochs_list=self.epochs)#[5,5,2]

        for i in range(1,n_iters):
            y_prior, theta_prior, _ = self.y_sampler.generate_data3(sample_size, i, self.y_dim1, prior=theta_posterior)
            self.y_true = y_prior[0:1,:]
            data_y1 = y_prior
            data_y2 = theta_prior
            self.sess.run(tf.global_variables_initializer())
            theta_posterior = self.train(data_y1, data_y2, i, epochs_list=self.epochs)
            
    
    def train(self,data_y1,data_y2,iteration,epochs_list):
        epochs = epochs_list[0]
        batch_size = self.batch_size
        counter = 1
        start_time = time.time()
        #STAGE 1 train RTM(x1,y1)
        for epoch in range(epochs):
            #to modify shuffle together
            train_idx = list(range(data_y1.shape[0]))
            random.shuffle(train_idx)
            data_y1 = data_y1[train_idx]
            data_y2 = data_y2[train_idx]
            lr = 2e-4 if epoch < epochs/2 else 2e-4*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = len(data_y1) // batch_size
            #lr decay, add later
            for idx in range(batch_idxs):
                #bx = data_x[batch_size*idx:batch_size*(idx+1)]
                bx1 = self.x_sampler1.get_batch(batch_size)
                bx2 = self.x_sampler2.get_batch(batch_size)
                #bx = self.x_sampler.train(batch_size)
                by1 = data_y1[batch_size*idx:batch_size*(idx+1)]
                by2 = data_y2[batch_size*idx:batch_size*(idx+1)]
                #update G and get generated fake data
                fake_bx, fake_by, g_summary, _ = self.sess.run([self.x_,self.y_,self.g_merged_summary ,self.g_h_optim1], feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2, self.lr:lr})
                self.summary_writer.add_summary(g_summary,counter)
                #random choose one batch from the previous 50 batches as fake batch
                [fake_bx,fake_by] = self.pool([fake_bx,fake_by])
                fake_bx1 = fake_bx[:,:self.x_dim1]
                fake_bx2 = fake_bx[:,self.x_dim1:]
                fake_by1 = fake_by[:,:self.y_dim1]
                fake_by2 = fake_by[:,self.y_dim1:]
                #update D
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim1], feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2, \
                    self.fake_x1: fake_bx1,self.fake_x2: fake_bx2, self.fake_y1: fake_by1,self.fake_y2: fake_by2,self.lr:lr})
                self.summary_writer.add_summary(d_summary,counter)
                #quick test on a random batch data
                if counter % 100 == 0:
                    bx1 = self.x_sampler1.train(batch_size)
                    bx2 = self.x_sampler2.train(batch_size)
                    idx = np.random.randint(low = 0, high = data_y1.shape[0], size = batch_size)
                    by1,by2 = data_y1[idx],data_y2[idx]
                    #by1,by2 = self.y_sampler.train(batch_size)

                    g_loss_adv1, h_loss_adv1, l2_loss_x1, l2_loss_y1, g_loss1, \
                        h_loss1, g_h_loss1, fake_bx1, fake_bx2, fake_by1, fake_by2 = self.sess.run(
                        [self.g_loss_adv1, self.h_loss_adv1, self.l2_loss_x1, self.l2_loss_y1, \
                        self.g_loss1, self.h_loss1, self.g_h_loss1, self.x1_, self.x2_, self.y1_,self.y2_],
                        feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2}
                    )
                    dx_loss1, dy_loss1, d_loss1 = self.sess.run([self.dx_loss1, self.dy_loss1, self.d_loss1], \
                        feed_dict={self.x1: bx1, self.x2: bx2, self.y1: by1,self.y2: by2, self.fake_x1: fake_bx1,self.fake_x2: fake_bx2,\
                             self.fake_y1: fake_by1,self.fake_y2: fake_by2})

                    print('Round [%d] Epoch [%d] Iter [%d] Time [%5.4f] g_loss_adv1 [%.4f] h_loss_adv1 [%.4f] l2_loss_x1 [%.4f] \
                        l2_loss_y1 [%.4f] g_loss1 [%.4f] h_loss1 [%.4f] g_h_loss1 [%.4f] dx_loss1 [%.4f] \
                        dy_loss1 [%.4f] d_loss1 [%.4f]' %
                        (iteration, epoch, counter, time.time() - start_time, g_loss_adv1, h_loss_adv1, l2_loss_x1, l2_loss_y1, \
                        g_loss1, h_loss1, g_h_loss1, dx_loss1, dy_loss1, d_loss1))                 
                counter+=1
            if (epoch+1)%5==0 or epoch+1==epochs:
                sample_size = data_y2.shape[0]
                x1 = self.x_sampler1.get_batch(sample_size)
                x2 = self.x_sampler2.get_batch(sample_size)
                y1_, y2_ = self.predict_y(x1,x2)
                np.savez('%s/stage1_RTM_iter%d_epoch%d.npz'%(self.save_dir,iteration,epoch),y1_,y2_)


        
        #STAGE 2 pretrain GAN(x2,y2)
        epochs = epochs_list[1]
        for epoch in range(epochs):
            train_idx = list(range(data_y1.shape[0]))
            random.shuffle(train_idx)
            data_y1 = data_y1[train_idx]
            data_y2 = data_y2[train_idx]
            lr = 1e-4 #if epoch < epochs/2 else 2e-4*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = len(data_y1) // batch_size
            #lr decay, add later
            for idx in range(batch_idxs):
                #bx = data_x[batch_size*idx:batch_size*(idx+1)]
                bx1 = self.x_sampler1.get_batch(batch_size)
                bx2 = self.x_sampler2.get_batch(batch_size)
                #bx = self.x_sampler.train(batch_size)
                by1 = data_y1[batch_size*idx:batch_size*(idx+1)]
                by2 = data_y2[batch_size*idx:batch_size*(idx+1)]
                #update D
                _ = self.sess.run(self.d_optim2, feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2, self.lr:lr})
                #update G 
                _ = self.sess.run(self.g_optim2, feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2, self.lr:lr})

                #self.summary_writer.add_summary(g_summary,counter)

                #quick test on a random batch data
                if counter % 100 == 0:
                    bx1 = self.x_sampler1.train(batch_size)
                    bx2 = self.x_sampler2.train(batch_size)
                    idx = np.random.randint(low = 0, high = data_y1.shape[0], size = batch_size)
                    by1,by2 = data_y1[idx],data_y2[idx]

                    g_loss_adv2, dy_loss2 = self.sess.run([self.g_loss_adv2, self.dy_loss2],
                        feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2}
                    )
                    print('Round [%d] Epoch [%d] Iter [%d] Time [%.4f] g_loss_adv2 [%.4f] dy_loss2 [%.4f]'%
                        (iteration, epoch, counter, time.time() - start_time, g_loss_adv2, dy_loss2))                 
                counter+=1
            if (epoch+1)%5==0 or epoch+1==epochs:
                #save pretrain theta
                sample_size = data_y2.shape[0]
                x1 = self.x_sampler1.get_batch(sample_size)
                x2 = self.x_sampler2.get_batch(sample_size)
                y1_, y2_ = self.predict_y(x1,x2)
                np.savez('%s/stage2_pretrain_iter%d_epoch%d.npz'%(self.save_dir,iteration,epoch),y1_,y2_)
                


        #STAGE 3 train GAN((x1,x2),(y1,y2))
        epochs = epochs_list[2]
        for epoch in range(epochs):
            train_idx = list(range(data_y1.shape[0]))
            random.shuffle(train_idx)
            data_y1 = data_y1[train_idx]
            data_y2 = data_y2[train_idx]
            lr = 1e-4 #if epoch < epochs/2 else 2e-4*float(epochs-epoch)/float(epochs-epochs/2)
            batch_idxs = len(data_y1) // batch_size
            #lr decay, add later
            for idx in range(batch_idxs):
                #bx = data_x[batch_size*idx:batch_size*(idx+1)]
                bx1 = self.x_sampler1.get_batch(batch_size)
                bx2 = self.x_sampler2.get_batch(batch_size)
                #bx = self.x_sampler.train(batch_size)
                by1 = data_y1[batch_size*idx:batch_size*(idx+1)]
                by2 = data_y2[batch_size*idx:batch_size*(idx+1)]
                #update D
                _ = self.sess.run(self.d_optim, feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2, self.lr:lr})
                #update G 
                _ = self.sess.run(self.g_optim, feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2, self.lr:lr})

                #self.summary_writer.add_summary(g_summary,counter)

                #quick test on a random batch data
                if counter % 100 == 0:
                    bx1 = self.x_sampler1.train(batch_size)
                    bx2 = self.x_sampler2.train(batch_size)
                    idx = np.random.randint(low = 0, high = data_y1.shape[0], size = batch_size)
                    by1,by2 = data_y1[idx],data_y2[idx]

                    g_loss_adv, dy_loss = self.sess.run([self.g_loss_adv, self.dy_loss],
                        feed_dict={self.x1: bx1, self.x2:bx2, self.y1: by1, self.y2: by2}
                    )
                    print('Round [%d] Epoch [%d] Iter [%d] Time [%.4f] g_loss_adv [%.4f] dy_loss [%.4f]'%
                        (iteration, epoch, counter, time.time() - start_time, g_loss_adv, dy_loss))       
                counter+=1

            if (epoch+1)%2==0 or epoch+1==epochs:
                sample_size = data_y2.shape[0]
                data_x1_, _ = self.predict_x(self.y_true,data_y2[0:1,:])
                x1 = np.tile(data_x1_[0,:],(sample_size,1))
                x2 = self.x_sampler2.get_batch(sample_size)
                _, y2_ = self.predict_y(x1,x2)
                #y2_[:,1] %= np.pi #restrict [0,np.pi]
                np.save('%s/theta_posterior_iter%d_epoch%d.npy'%(self.save_dir,iteration,epoch),y2_)
        return y2_
 

    #predict with y_=G(x)
    def predict_y(self, x1, x2, bs=256):
        assert x1.shape[0] == x2.shape[0]
        N = x1.shape[0]
        y1_pred = np.zeros(shape=(N, self.y_dim1)) 
        y2_pred = np.zeros(shape=(N, self.y_dim2)) 
        for b in range(int(np.ceil(N*1.0 / bs))):

            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x1 = x1[ind, :]
            batch_x2 = x2[ind, :]
            batch_y1_, batch_y2_ = self.sess.run([self.y1_,self.y2_], feed_dict={self.x1:batch_x1,self.x2:batch_x2})
            y1_pred[ind, :] = batch_y1_
            y2_pred[ind, :] = batch_y2_
        return y1_pred, y2_pred
    
    #predict with x_=H(y)
    def predict_x(self,y1,y2,bs=256):
        assert y1.shape[0] == y2.shape[0]
        N = y1.shape[0]
        x1_pred = np.zeros(shape=(N, self.x_dim1)) 
        x2_pred = np.zeros(shape=(N, self.x_dim2)) 
        for b in range(int(np.ceil(N*1.0 / bs))):

            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y1 = y1[ind, :]
            batch_y2 = y2[ind, :]
            batch_x1_,batch_x2_  = self.sess.run([self.x1_, self.x2_], feed_dict={self.y1:batch_y1,self.y2:batch_y2})
            x1_pred[ind, :] = batch_x1_
            x2_pred[ind, :] = batch_x2_
        return x1_pred,x2_pred


    def evaluate(self,epoch,stage=1):
        data_x1, _ = self.x_sampler1.load_all()
        data_x2, _ = self.x_sampler2.load_all()
        #data_x2_fake = np.zeros(data_x2.shape)
        data_y1,data_y2 = self.y_sampler.load_all()
        data_x1_,data_x2_ = self.predict_x(data_y1,data_y2)
        data_y1_,data_y2_ = self.predict_y(data_x1,data_x2)
        #data_y1_fake,data_y2_fake = self.predict_y(data_x1,data_x2_fake)
        np.savez('{}/data_at_{}_stage_{}.npz'.format(self.save_dir, epoch, stage),data_x1,data_x2,data_y1,data_y2,data_x1_,data_x2_,data_y1_,data_y2_)
        #self.plot_density_2D([data_x,data_y,data_x_,data_y_], '%s/figs'%self.save_dir, epoch)

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
    parser.add_argument('--dx1', type=int, default=10)
    parser.add_argument('--dx2', type=int, default=10)
    parser.add_argument('--dy1', type=int, default=10)
    parser.add_argument('--dy2', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--epochs', nargs='+', type=int, help='set epochs for 3 stages')
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    x_dim1 = args.dx1
    x_dim2 = args.dx2
    y_dim1 = args.dy1
    y_dim2 = args.dy2
    batch_size = args.bs
    alpha = args.alpha
    beta = args.beta
    timestamp = args.timestamp
    epochs = args.epochs


    #g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=6,nb_units=256)
    #h_net = model.Generator(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=6,nb_units=256)
    g_net = model.Generator_PCN(input_dim1=x_dim1,input_dim2=x_dim2,output_dim1=y_dim1,output_dim2=y_dim2,name='g_net',nb_layers=3,nb_units=32)
    h_net = model.Generator_PCN(input_dim1=y_dim1,input_dim2=y_dim2,output_dim1=x_dim1,output_dim2=x_dim2,name='h_net',nb_layers=3,nb_units=32)
    dx_net1 = model.Discriminator(input_dim=x_dim1,name='dx_net1',nb_layers=2,nb_units=16)
    dy_net1 = model.Discriminator(input_dim=y_dim1,name='dy_net1',nb_layers=2,nb_units=16)
    dy_net2 = model.Discriminator(input_dim=y_dim2,name='dy_net2',nb_layers=2,nb_units=16)
    dy_net = model.Discriminator(input_dim=y_dim1+y_dim2,name='dx_net_all',nb_layers=2,nb_units=16)

    #xs = util.Y_sampler(N=10000, n_components=2,dim=y_dim,mean=3.0,sd=0.5)
    #ys = util.Y_sampler(N=10000, n_components=2,dim=x_dim,mean=0.0,sd=1)
    xs1 = util.Gaussian_sampler(N=10000,mean=np.zeros(x_dim1),sd=1.0)
    xs2 = util.Gaussian_sampler(N=10000,mean=np.zeros(x_dim2),sd=1.0)
    #xs = util.Gaussian_sampler(N=10000,mean=np.zeros(x_dim),sd=1.0)
    #xs = util.X_sampler(N=10000, dim=x_dim, mean=3.0)
    #ys = util.Gaussian_sampler(N=10000,mean=np.zeros(y_dim),sd=1.0)
    #ys = util.Bayesian_sampler(N=10000,dim1=y_dim1, dim2=y_dim2)
    #ys = util.Bayesian_sampler(N=5205,dim1=y_dim1, dim2=y_dim2)
    #ys = util.SV_sampler([0.0314, 0.9967, 0.0107, 19.6797, -1.1528],10)
    ys = util.Cosine_sampler([1. / 80, np.pi / 4, 0, np.log(2)])
    pool = util.DataPool()
    
    #sys.exit()
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

    RTM = RoundtripModel(g_net, h_net, dx_net1, dy_net1, dy_net2, dy_net, xs1, xs2, ys, pool, batch_size, alpha, beta, epochs)

    if args.train == 'True':
        RTM.bayesian_iteration()
        #RTM.train()
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = 2499)
            #RTM.estimate_py_with_IS(2999)
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            from scipy.stats import norm
            from scipy.stats import pearsonr
            t0=time.time()
            y_val,_ = ys.resample(1000)
            start,end,nb_intervals = 0.2, 1.5, 15
            def cal_pdf(y_point,mean):
                y_pdf=[]
                for i in range(len(y_point)):
                    y_pdf.append(np.mean(norm.pdf(mean[:,i], loc=y_point[i], scale=0.5)))
                return np.prod(y_pdf)
            gmm_mean = ys.mean
            py_true = map(cal_pdf,y_val,[gmm_mean]*len(y_val))
            mle=[]
            for sd in np.linspace(start,end,nb_intervals):
            #for sd in [0.757]:
                #for scale in np.linspace(0.2,2,10):
                for scale in [0.5]:
                    py_est = RTM.estimate_py_with_IS(y_val,2999,sd_y=sd,scale=scale,sample_size=10000,save=False)
                    print 'RTM', sd,scale, pearsonr(py_true,py_est)[0]
                mle.append(np.sum(np.log(py_est+1e-20)))
            print mle
            plt.plot(np.linspace(start,end,nb_intervals),mle)
            plt.xlabel('sd')
            plt.ylabel('log-likelihood')
            plt.savefig('model_select_at_dim%d.png'%(gmm_mean.shape[1]))
            best_sd = np.linspace(start,end,nb_intervals)[mle.index(np.max(mle))]
            print('%.2f seconds'%(time.time()-t0))
            print('Best sd for RTM:%.2f'%best_sd)
