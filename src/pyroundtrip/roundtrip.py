import tensorflow as tf
from .model import *
import numpy as np
from .util import *
import dateutil.tz
import datetime
import os, sys

class Roundtrip(object):
    """ Roundtrip model for density estimation.
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(Roundtrip, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = Discriminator(input_dim=params['x_dim'],model_name='dx_net',
                                        nb_units=params['dx_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.initilize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, self.params['z_dim'])))
        self.e_net(np.zeros((1, self.params['x_dim'])))
        self.dz_net(np.zeros((1, self.params['z_dim'])))
        self.dx_net(np.zeros((1, self.params['x_dim'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.dx_net.summary())

    @tf.function
    def train_gen_step(self, data_z, data_x):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: date tensor with shape [batch_size, x_dim].
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            data_x_ = self.g_net(data_z)
            data_z_ = self.e_net(data_x)

            data_z__= self.e_net(data_x_)
            data_x__ = self.g_net(data_z_)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            #g_loss_adv = -tf.reduce_mean(data_dx_)
            #e_loss_adv = -tf.reduce_mean(data_dz_)
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = g_loss_adv+e_loss_adv+self.params['alpha']*(l2_loss_x + l2_loss_z)

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))
        return g_loss_adv, e_loss_adv, l2_loss_x, l2_loss_z, g_e_loss

    @tf.function
    def train_disc_step(self, data_z, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: data tensor with shape [batch_size, x_dim].
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            data_x_ = self.g_net(data_z)
            data_z_ = self.e_net(data_x)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            #dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            #dx_loss = -tf.reduce_mean(data_dx) + tf.reduce_mean(data_dx_)
            dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,v_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss = dx_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + gpx_loss)

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)

        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        return dx_loss, dz_loss, d_loss

    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500,
            startoff=0, verbose=1, save_format='txt'):
        f_params = open('{}/params.txt'.format(self.save_dir),'w')
        f_params.write(str(self.params))
        f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)

        for batch_idx in range(n_iter+1):
            for _ in range(self.params['g_d_freq']):
                batch_x = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(len(batch_x))
                dx_loss, dz_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            batch_x = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(len(batch_x))
            g_loss_adv, e_loss_adv, l2_loss_x, l2_loss_z, g_e_loss = self.train_gen_step(batch_z, batch_x)

            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : g_loss_adv [%.4f], e_loss_adv [%.4f],\
                l2_loss_x [%.4f], l2_loss_z [%.4f], g_e_loss [%.4f], dx_loss [%.4f], dz_loss [%.4f], d_loss [%.4f]''' \
                %(batch_idx, g_loss_adv, e_loss_adv, l2_loss_x, l2_loss_z, g_e_loss,
                dx_loss, dz_loss, d_loss)
                if verbose:
                    print(loss_contents)
                #px_est = self.estimate_px_with_CF(self.data_sampler.load_all(),sd_x=self.params['sd_x'],log=True)
                self.evaluate(batch_idx)
                px_est = self.estimate_px_with_IS(self.data_sampler.load_all(),
                                                    sd_x=self.params['sd_x'],
                                                    scale=self.params['scale'],
                                                    sample_size=self.params['sample_size'])
                self.save('{}/px_est_{}.{}'.format(self.save_dir, batch_idx, save_format), px_est)

                if self.params['save_model']:
                    ckpt_save_path = self.ckpt_manager.save(batch_idx)
                    #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))

    def evaluate(self, batch_idx, n=100):
        data_z_ = self.e_net(self.data_sampler.load_all())
        np.save('{}/data_z_at_{}.npy'.format(self.save_dir,batch_idx),data_z_)
        if self.params['dataset'] == 'indep_gmm':
            v1, v2, data_grid = create_2d_grid_data(x1_min=-1.5, x1_max=1.5, x2_min=-1.5, x2_max=1.5, n=n)
            px_est = self.estimate_px_with_IS(data_grid,sd_x=self.params['sd_x'],scale=self.params['scale'],sample_size=self.params['sample_size'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_IS_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
            px_est = self.estimate_px_with_CF(data_grid,sd_x=self.params['sd_x'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_CF_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
        elif self.params['dataset'] == 'involute':
            v1, v2, data_grid = create_2d_grid_data(x1_min=-6, x1_max=5, x2_min=-5, x2_max=5, n=n)
            px_est = self.estimate_px_with_IS(data_grid,sd_x=self.params['sd_x'],scale=self.params['scale'],sample_size=self.params['sample_size'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_IS_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
            px_est = self.estimate_px_with_CF(data_grid,sd_x=self.params['sd_x'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_CF_density_pre_at_{}.png'.format(self.save_dir,batch_idx))

    #estimate pdf of x (e.g., p(x)) with importance sampling
    def estimate_px_with_IS(self,x_points, sd_x=0.5, scale=0.5,sample_size=40000,bs=1024,df = 1, log=True):
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

        def px_given_z(zip_list):
            '''
            calculate p(x|z)
            z_points with shape (sample_size, z_dim)
            x_point wish shape (x_dim, )
            '''
            z_points = zip_list[0]
            x_point = zip_list[1]
            x_points_ = self.g_net(z_points).numpy()
            if log:
                return -self.params['x_dim']*np.log((np.sqrt(2*np.pi)*sd_x))-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2)
            else:
                return 1. / ((np.sqrt(2*np.pi)*sd_x)**self.params['x_dim']) * np.exp(-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2))

        def w_likelihood_ratio(zip_list):
            '''
            calculate w=pz/px
            z_point with shape (z_dim, )
            z_points with shape (sample_size,z_dim)
            '''
            z_point = zip_list[0]
            z_points = zip_list[1]
            if log:
                log_qz = np.sum(t.logpdf(z_point-z_points,df,loc=0,scale=scale),axis=1)
                log_pz = -self.params['z_dim']*np.log(np.sqrt(2*np.pi))-(np.sum((z_points)**2,axis=1))/2.
                return log_pz-log_qz
            else:
                qz =np.prod(t.pdf(z_point-z_points, df,loc=0,scale=scale),axis=1)
                pz = 1. / (np.sqrt(2*np.pi)**self.params['z_dim']) * np.exp(-(np.sum((z_points)**2,axis=1))/2.)
                return pz / qz

        #sample a set of points given each z_point from importance distribution
        def sample_from_qz(z_point):
            '''
            multivariate student t distribution can be constructed from a multivariate Gaussian 
            one can also use t.rvs to sample (see the uncommented line) which is lower
            '''
            S = np.diag(scale**2 * np.ones(self.params['z_dim']))
            z1 = np.random.chisquare(df, sample_size)/df
            z2 = np.random.multivariate_normal(np.zeros(self.params['z_dim']),S,(sample_size,))
            return z_point + z2/np.sqrt(z1)[:,None]
            #return np.hstack([t.rvs(df, loc=value, scale=scale, size=(sample_size,1), random_state=None) for value in z_point])
        z_points_ = self.e_net(x_points).numpy()
        N = len(x_points)
        px_given_z_list=[]
        w_likelihood_ratio_list=[]
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               #ind = tf.range(b*bs, N)
               ind = np.arange(b*bs, N, dtype='int32')
            else:
               #ind = tf.range(b*bs, (b+1)*bs,)
               ind = np.arange(b*bs, (b+1)*bs, dtype='int32')
            batch_x_points = x_points[ind, :]
            batch_z_points_ = z_points_[ind, :]
            batch_z_points_sample_list = program_paral(sample_from_qz,batch_z_points_)
            batch_px_given_z_list = program_paral(px_given_z, zip(batch_z_points_sample_list, batch_x_points))
            batch_w_likelihood_ratio_list = program_paral(w_likelihood_ratio, zip(batch_z_points_, batch_z_points_sample_list))
            px_given_z_list += batch_px_given_z_list
            w_likelihood_ratio_list += batch_w_likelihood_ratio_list

        #calculate p(x)=int(p(x|z)*p(z)dz)=int(p(x|z)*w(z)q(z)dz)=E(p(x|z)*w(z)) where z~q(z)
        if log:
            px_list = list(map(lambda z, x: z+x,px_given_z_list,w_likelihood_ratio_list))
            max_idx_list = [np.where(item==max(item))[0][0] for item in px_list]
            px_est = np.array([np.log(np.sum(np.exp(item[0]-item[0][item[1]])))-np.log(sample_size)+item[0][item[1]] for item in zip(px_list,max_idx_list)])
        else:
            px_list = list(map(lambda z, x: z*x,px_given_z_list,w_likelihood_ratio_list))
            px_est = np.array([np.mean(item) for item in px_list])
        return px_est

    @tf.function
    def get_jacobian(self, data_z):
        """get jacobian matrix.
        Args:
            input: a tensor with shape [batch_size, z_dim].
        Returns:
            returns batch jacobian with shape [batch_size, x_dim, z_dim]
        """  
        with tf.GradientTape() as j_tape:
            j_tape.watch(data_z)
            data_x_ = self.g_net(data_z)
        batch_jacobian = j_tape.batch_jacobian(data_x_, data_z)
        return batch_jacobian

    #estimate pdf of x (e.g., p(x)) with Laplace approximation (closed-from)
    def estimate_px_with_CF(self,x_points,sd_x=0.5,log=True):
        from scipy.stats import t
        from multiprocessing.dummy import Pool as ThreadPool

        #multi-process to parallel the program
        def program_paral(func,param_list):
            pool = ThreadPool()
            results = pool.map(func,param_list)
            pool.close()
            pool.join()
            return results

        z_points_ = self.e_net(x_points)
        x_points__ = self.g_net(z_points_)

        rt_error = np.sum((x_points-x_points__)**2,axis=1)

        #get jocobian matrix with shape (N, x_dim, z_dim)
        jacob_mat = self.get_jacobian(z_points_)
        #jocobian matrix transpose with shape (N, z_dim, x_dim)
        jacob_mat_transpose = tf.transpose(jacob_mat,perm=[0, 2, 1])

        #matrix A = G^T(z_)*G(z_) with shape (N, z_dim, z_dim)
        A = list(map(lambda x, y: np.dot(x,y), jacob_mat_transpose, jacob_mat))

        #vector b = grad_^T(G(z_))*(x-x__) with shape (N, z_dim)
        b = list(map(lambda x, y: np.dot(x,y), jacob_mat_transpose, x_points-x_points__))

        #covariant matrix in constructed multivariate Gaussian with shape (N, z_dim, z_dim)
        Sigma = list(map(lambda x: np.linalg.inv(np.eye(self.params['z_dim'])+x/sd_x**2),A))
        Sigma_inv = list(map(lambda x: np.eye(self.params['z_dim'])+x/sd_x**2,A))

        #mean vector in constructed multivariate Gaussian with shape (N, z_dim)
        mu = list(map(lambda x,y,z: x.dot(y/sd_x**2-z),Sigma,b,z_points_))

        #constant term c(x) in the integral c(x) = l2_norm(z_)^2 + l2_norm(x-x__)^2/sigma**2-mu^T*Sigma*mu
        c_x = list(map(lambda x,y,z,w: np.sum(x**2)+y/sd_x**2-z.T.dot(w).dot(z), z_points_, rt_error, mu, Sigma_inv))

        if log:
            px_est = list(map(lambda x,y:-self.params['x_dim']*np.log(np.sqrt(2*np.pi)*sd_x)+0.5*np.log(np.linalg.det(x))-0.5*y, Sigma, c_x))
        else:
            px_est = list(map(lambda x,y: 1./(np.sqrt(2*np.pi)*sd_x)**self.params['x_dim']* sd_x**self.params['x_dim'] *np.sqrt(np.linalg.det(x)) * np.exp(-0.5*y), Sigma, c_x))
        return np.array(px_est)

    def save(self, fname, data):
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()


class VariationalRoundtrip(object):
    """ Roundtrip model with 
        1) variational inference in latent space.
        2) trainable variance in data space
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(VariationalRoundtrip, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = 1+params['x_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = 2*params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = Discriminator(input_dim=params['x_dim'],model_name='dx_net',
                                        nb_units=params['dx_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.initilize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, self.params['z_dim'])))
        self.e_net(np.zeros((1, self.params['x_dim'])))
        self.dz_net(np.zeros((1, self.params['z_dim'])))
        self.dx_net(np.zeros((1, self.params['x_dim'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.dx_net.summary())

    @tf.function
    def train_gen_step(self, data_z, data_x):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: date tensor with shape [batch_size, x_dim].
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            x_mean, x_logvar = self.decode(data_z)
            data_x_ = self.reparameterize(x_mean, x_logvar)

            z_mean, z_logvar = self.encode(data_x)
            data_z_ = self.reparameterize(z_mean, z_logvar)

            z_mean_, z_logvar_ = self.encode(data_x_)
            data_z__ = self.reparameterize(z_mean_, z_logvar_)

            x_mean_, x_logvar_ = self.decode(data_z_)
            data_x__ = self.reparameterize(x_mean_, x_logvar_)

            # variational loss
            logpx_z = -tf.reduce_mean((data_x - data_x__)**2,axis=1)
            logqz_x = self.log_normal_pdf(data_z_, z_mean, z_logvar)
            logpz = self.log_normal_pdf(data_z_, 0., 0.)
            kl_loss = logqz_x-logpz # here it is not the formula of KL_loss, so will result in negative values(similar)
            elbo = tf.reduce_mean(logpx_z - kl_loss)

            data_dx_ = self.dx_net(data_x_)
            #data_dz_ = self.dz_net(data_z_)
            
            #l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            #e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = -elbo + g_loss_adv + self.params['alpha']*l2_loss_z

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))
        return tf.reduce_mean(kl_loss), elbo, g_loss_adv, l2_loss_z, g_e_loss

    @tf.function
    def train_disc_step(self, data_z, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: data tensor with shape [batch_size, x_dim].
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            x_mean, x_logvar = self.decode(data_z)
            data_x_ = self.reparameterize(x_mean, x_logvar)

            # z_mean, z_logvar = self.encode(data_x)
            # data_z_ = self.reparameterize(z_mean, z_logvar)
            
            data_dx_ = self.dx_net(data_x_)
            #data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            #data_dz = self.dz_net(data_z)
            
            #dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
            #    +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            #gradient penalty for z
            # data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            # data_dz_hat = self.dz_net(data_z_hat)
            # grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            # grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            # gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,v_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss =  dx_loss + self.params['gamma']*gpx_loss

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        return dx_loss, d_loss

    def encode(self, x):
        mean, logvar = tf.split(self.e_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z):
        output = self.g_net(z)
        mean = output[:, :self.params['x_dim']]
        logvar = output[:, self.params['x_dim']:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def log_normal_pdf(self, sample, mean, logvar, axis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=axis)

    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500,
            startoff=0, verbose=1, save_format='txt'):
        f_params = open('{}/params.txt'.format(self.save_dir),'w')
        f_params.write(str(self.params))
        #f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)

        for batch_idx in range(n_iter+1):
            for _ in range(self.params['g_d_freq']):
                batch_x = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(len(batch_x))
                dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            batch_x = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(len(batch_x))
            kl_loss, elbo, g_loss_adv, l2_loss_z, g_e_loss = self.train_gen_step(batch_z, batch_x)

            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : kl_loss [%.4f], elbo [%.4f], g_loss_adv [%.4f],\
                l2_loss_z [%.4f], g_e_loss [%.4f], dx_loss [%.4f], d_loss [%.4f]''' \
                %(batch_idx, kl_loss, elbo, g_loss_adv, l2_loss_z, g_e_loss, dx_loss, d_loss)
                if verbose:
                    print(loss_contents)
                f_params.write(loss_contents+'\n')
                # px_est = self.estimate_px_with_CF(self.data_sampler.load_all(),log=True)
                # self.evaluate(batch_idx)
                # px_est = self.estimate_px_with_IS(self.data_sampler.X_test,
                #                                     sd_x=self.params['sd_x'],
                #                                     scale=self.params['scale'],
                #                                     sample_size=self.params['sample_size'],log=False)
                # px_true = self.data_sampler.get_density(self.data_sampler.X_test)
                # print('Batch [%d]: Pearson correlation is %.3f, Spearman correlation is %.3f'%(batch_idx, pearsonr(px_est,px_true)[0],
                #         spearmanr(px_est,px_true)[0]))
                
                #self.save('{}/px_est_{}.{}'.format(self.save_dir, batch_idx, save_format), px_est)
                if self.params['save_model']:
                    ckpt_save_path = self.ckpt_manager.save(batch_idx)
                    #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))
        f_params.close()
    def evaluate(self, batch_idx, n=100):
        #data_z_ = self.e_net(self.data_sampler.load_all())
        #np.save('{}/data_z_at_{}.npy'.format(self.save_dir,batch_idx),data_z_)

        if self.params['dataset'] == 'indep_gmm':
            v1, v2, data_grid = create_2d_grid_data(x1_min=-1.5, x1_max=1.5, x2_min=-1.5, x2_max=1.5, n=n)
            px_est = self.estimate_px_with_IS(data_grid,sd_x=self.params['sd_x'],scale=self.params['scale'],sample_size=self.params['sample_size'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_IS_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
            #px_est = self.estimate_px_with_CF(data_grid,log=False)
            #px_est = px_est.reshape((n,n))
            #plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_CF_density_pre_at_{}.png'.format(self.save_dir,batch_idx))

        elif self.params['dataset'] == 'involute':
            v1, v2, data_grid = create_2d_grid_data(x1_min=-6, x1_max=5, x2_min=-5, x2_max=5, n=n)
            px_est = self.estimate_px_with_IS(data_grid,sd_x=self.params['sd_x'],scale=self.params['scale'],sample_size=self.params['sample_size'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_IS_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
            # px_est = self.estimate_px_with_CF(data_grid,log=False)
            # px_est = px_est.reshape((n,n))
            # plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_CF_density_pre_at_{}.png'.format(self.save_dir,batch_idx))

    #estimate pdf of x (e.g., p(x)) with importance sampling
    def estimate_px_with_IS(self,x_points, sd_x=0.5, scale=0.5,sample_size=40000,bs=1024,df = 1, log=True):
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

        def px_given_z(zip_list):
            '''
            calculate p(x|z)
            z_points with shape (sample_size, z_dim)
            x_point wish shape (x_dim, )
            '''
            z_points = zip_list[0]
            x_point = zip_list[1]
            #x_points_ = self.g_net(z_points).numpy()
            mean, logvar = self.decode(z_points)
            x_points_ = self.reparameterize(mean, logvar).numpy()
            if log:
                return -self.params['x_dim']*np.log((np.sqrt(2*np.pi)*sd_x))-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2)
            else:
                return 1. / ((np.sqrt(2*np.pi)*sd_x)**self.params['x_dim']) * np.exp(-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2))

        def w_likelihood_ratio(zip_list):
            '''
            calculate w=pz/px
            z_point with shape (z_dim, )
            z_points with shape (sample_size,z_dim)
            '''
            z_point = zip_list[0]
            z_points = zip_list[1]
            if log:
                log_qz = np.sum(t.logpdf(z_point-z_points,df,loc=0,scale=scale),axis=1)
                log_pz = -self.params['z_dim']*np.log(np.sqrt(2*np.pi))-(np.sum((z_points)**2,axis=1))/2.
                return log_pz-log_qz
            else:
                qz =np.prod(t.pdf(z_point-z_points, df,loc=0,scale=scale),axis=1)
                pz = 1. / (np.sqrt(2*np.pi)**self.params['z_dim']) * np.exp(-(np.sum((z_points)**2,axis=1))/2.)
                return pz / qz

        #sample a set of points given each z_point from importance distribution
        def sample_from_qz(z_point):
            '''
            multivariate student t distribution can be constructed from a multivariate Gaussian 
            one can also use t.rvs to sample (see the uncommented line) which is lower
            '''
            S = np.diag(scale**2 * np.ones(self.params['z_dim']))
            z1 = np.random.chisquare(df, sample_size)/df
            z2 = np.random.multivariate_normal(np.zeros(self.params['z_dim']),S,(sample_size,))
            return z_point + z2/np.sqrt(z1)[:,None]
            #return np.hstack([t.rvs(df, loc=value, scale=scale, size=(sample_size,1), random_state=None) for value in z_point])
        z_mean, z_logvar = self.encode(x_points)
        z_points_ = self.reparameterize(z_mean, z_logvar).numpy()
        #z_points_ = self.e_net(x_points).numpy()
        N = len(x_points)
        px_given_z_list=[]
        w_likelihood_ratio_list=[]
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               #ind = tf.range(b*bs, N)
               ind = np.arange(b*bs, N, dtype='int32')
            else:
               #ind = tf.range(b*bs, (b+1)*bs,)
               ind = np.arange(b*bs, (b+1)*bs, dtype='int32')
            batch_x_points = x_points[ind, :]
            batch_z_points_ = z_points_[ind, :]
            batch_z_points_sample_list = program_paral(sample_from_qz,batch_z_points_)
            batch_px_given_z_list = program_paral(px_given_z, zip(batch_z_points_sample_list, batch_x_points))
            batch_w_likelihood_ratio_list = program_paral(w_likelihood_ratio, zip(batch_z_points_, batch_z_points_sample_list))
            px_given_z_list += batch_px_given_z_list
            w_likelihood_ratio_list += batch_w_likelihood_ratio_list

        #calculate p(x)=int(p(x|z)*p(z)dz)=int(p(x|z)*w(z)q(z)dz)=E(p(x|z)*w(z)) where z~q(z)
        if log:
            px_list = list(map(lambda z, x: z+x,px_given_z_list,w_likelihood_ratio_list))
            max_idx_list = [np.where(item==max(item))[0][0] for item in px_list]
            px_est = np.array([np.log(np.sum(np.exp(item[0]-item[0][item[1]])))-np.log(sample_size)+item[0][item[1]] for item in zip(px_list,max_idx_list)])
        else:
            px_list = list(map(lambda z, x: z*x,px_given_z_list,w_likelihood_ratio_list))
            px_est = np.array([np.mean(item) for item in px_list])
        return px_est

    @tf.function
    def get_jacobian(self, data_z):
        """get jacobian matrix.
        Args:
            input: a tensor with shape [batch_size, z_dim].
        Returns:
            returns batch jacobian with shape [batch_size, x_dim, z_dim]
        """
        with tf.GradientTape() as j_tape:
            j_tape.watch(data_z)
            mean, logvar = tf.split(self.g_net(data_z), num_or_size_splits=2, axis=1)
        batch_mean_jacobian = j_tape.batch_jacobian(mean, data_z)
        #batch_logvar_jacobian = j_tape.batch_jacobian(logvar, data_z)
        return batch_mean_jacobian

    #estimate pdf of x (e.g., p(x)) with Laplace approximation (closed-from)
    #@tf.function
    def estimate_px_with_CF(self,x_points,log=True):
        z_points_ = self.e_net(x_points)
        #x_points__ = self.g_net(z_points_)
        mean_, logvar_ = self.decode(z_points_)
        x_points__ = self.reparameterize(mean_, logvar_)
        
        rt_error = list(map(lambda x,y: tf.squeeze(tf.matmul(tf.matmul(tf.expand_dims(x,0), tf.linalg.diag((tf.exp(-y)))), tf.expand_dims(x,1))),x_points-mean_,logvar_))
        #rt_error = np.sum((x_points-x_points__)**2,axis=1)

        #get jocobian matrix with shape (N, x_dim, z_dim)
        jacob_mat = self.get_jacobian(z_points_)
        #jocobian matrix transpose with shape (N, z_dim, x_dim)
        jacob_mat_transpose = tf.transpose(jacob_mat,perm=[0, 2, 1])

        #matrix A = mu^T(z_)*Sigma^(-1)*mu(z_) with shape (N, z_dim, z_dim)
        A = list(map(lambda x,y,z: tf.matmul(tf.matmul(x, tf.linalg.diag((tf.exp(-y)))), z), jacob_mat_transpose,logvar_, jacob_mat))

        #vector b = grad_^T(mu(z_))*Sigma^(-1)*(x-x__) with shape (N, z_dim)
        b = list(map(lambda x,y,z: tf.squeeze(tf.matmul(tf.matmul(x, tf.linalg.diag((tf.exp(-y)))), tf.expand_dims(z,1))), jacob_mat_transpose, logvar_, x_points-mean_))

        #covariant matrix in constructed multivariate Gaussian with shape (N, z_dim, z_dim)
        Sigma = list(map(lambda x: tf.linalg.inv(tf.eye(self.params['z_dim'])+x),A))
        Sigma_inv = list(map(lambda x: tf.eye(self.params['z_dim'])+x,A))

        #mean vector in constructed multivariate Gaussian with shape (N, z_dim)
        mu = list(map(lambda x,y,z: tf.squeeze(tf.matmul(x,tf.expand_dims(y-z,1))),Sigma,b,z_points_))

        #constant term c(x) in the integral c(x) = l2_norm(z_)^2 + l2_norm(x-x__)^2/sigma**2-mu^T*Sigma*mu
        c_x = list(map(lambda x,y,z,w: tf.reduce_sum(x**2)+y-tf.squeeze(tf.matmul(tf.matmul(tf.expand_dims(z,0),w),tf.expand_dims(z,1))), z_points_, rt_error, mu, Sigma_inv))

        if log:
            #px_est = list(map(lambda x,y:-self.params['x_dim']*np.log(np.sqrt(2*np.pi))+np.sum(logvar_)/2+0.5*np.log(np.linalg.det(x))-0.5*y, Sigma, c_x))
            px_est = list(map(lambda x,y:-self.params['x_dim']*np.log(np.sqrt(2*np.pi))+tf.reduce_sum(logvar_)/2+0.5*tf.math.log(tf.linalg.det(x))-0.5*y, Sigma, c_x))
        else:
            #px_est = list(map(lambda x,y: 1./(np.sqrt(2*np.pi))**self.params['x_dim']* np.prod(tf.exp(logvar_/2)) *np.sqrt(np.linalg.det(x)) * np.exp(-0.5*y), Sigma, c_x))
            px_est = list(map(lambda x,y: 1./(np.sqrt(2*np.pi))**self.params['x_dim']* tf.reduce_prod(tf.exp(logvar_/2)) *tf.math.sqrt(tf.linalg.det(x)) * tf.math.exp(-0.5*y), Sigma, c_x))
        return np.array(px_est)

    def save(self, fname, data):
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()


class RoundtripTV(object):
    """ Roundtrip model with 
        1) trainable variance in data space
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(RoundtripTV, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = 1+params['x_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = Discriminator(input_dim=params['x_dim'],model_name='dx_net',
                                        nb_units=params['dx_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.initilize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, self.params['z_dim'])))
        self.e_net(np.zeros((1, self.params['x_dim'])))
        self.dz_net(np.zeros((1, self.params['z_dim'])))
        self.dx_net(np.zeros((1, self.params['x_dim'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.dx_net.summary())

    @tf.function
    def train_gen_step(self, data_z, data_x):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: date tensor with shape [batch_size, x_dim].
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            x_mean, x_logvar = self.decode(data_z)
            data_x_ = self.reparameterize(x_mean, x_logvar)

            data_z_ = self.e_net(data_x)
            data_z__ = self.e_net(data_x_)

            x_mean_, x_logvar_ = self.decode(data_z_)
            data_x__ = self.reparameterize(x_mean_, x_logvar_)

            # variational loss
            # logpx_z = -tf.reduce_mean((data_x - data_x__)**2,axis=1)
            # logqz_x = self.log_normal_pdf(data_z_, z_mean, z_logvar)
            # logpz = self.log_normal_pdf(data_z_, 0., 0.)
            # kl_loss = logqz_x-logpz # here it is not the formula of KL_loss, so will result in negative values(similar)
            # elbo = tf.reduce_mean(logpx_z - kl_loss)

            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = e_loss_adv + g_loss_adv + self.params['alpha']*(l2_loss_z+l2_loss_x)

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))
        return e_loss_adv, g_loss_adv, l2_loss_z, l2_loss_x, g_e_loss

    @tf.function
    def train_disc_step(self, data_z, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: data tensor with shape [batch_size, x_dim].
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            x_mean, x_logvar = self.decode(data_z)
            data_x_ = self.reparameterize(x_mean, x_logvar)

            data_z_ = self.e_net(data_x)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,v_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss =  dx_loss + dz_loss + self.params['gamma']*(gpx_loss+gpz_loss)

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        return dz_loss, dx_loss, d_loss

    def encode(self, x):
        mean, logvar = tf.split(self.e_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z):
        output = self.g_net(z)
        mean = output[:, :self.params['x_dim']]
        logvar = output[:, self.params['x_dim']:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def log_normal_pdf(self, sample, mean, logvar, axis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=axis)

    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500,
            startoff=0, verbose=1, save_format='txt'):
        f_params = open('{}/params.txt'.format(self.save_dir),'w')
        f_params.write(str(self.params))
        #f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)

        for batch_idx in range(n_iter+1):
            for _ in range(self.params['g_d_freq']):
                batch_x = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(len(batch_x))
                dz_loss, dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            batch_x = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(len(batch_x))
            e_loss_adv, g_loss_adv, l2_loss_z, l2_loss_x, g_e_loss = self.train_gen_step(batch_z, batch_x)

            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : e_loss_adv [%.4f], g_loss_adv [%.4f], l2_loss_z [%.4f],\
                l2_loss_x [%.4f], g_e_loss [%.4f], dz_loss [%.4f], dx_loss [%.4f], d_loss [%.4f]''' \
                %(batch_idx, e_loss_adv, g_loss_adv, l2_loss_z, l2_loss_x, g_e_loss , dz_loss, dx_loss, d_loss)
                if verbose:
                    print(loss_contents)
                f_params.write(loss_contents+'\n')
                #px_est = self.estimate_px_with_CF(self.data_sampler.load_all(),log=True)
                self.evaluate(batch_idx)
                px_est = self.estimate_px_with_IS(self.data_sampler.load_all(),
                                                    sd_x=self.params['sd_x'],
                                                    scale=self.params['scale'],
                                                    sample_size=self.params['sample_size'])
                self.save('{}/px_est_{}.{}'.format(self.save_dir, batch_idx, save_format), px_est)

                if self.params['save_model']:
                    ckpt_save_path = self.ckpt_manager.save(batch_idx)
                    #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))
        f_params.close()
    def evaluate(self, batch_idx, n=100):
        #data_z_ = self.e_net(self.data_sampler.load_all())
        #np.save('{}/data_z_at_{}.npy'.format(self.save_dir,batch_idx),data_z_)

        if self.params['dataset'] == 'indep_gmm':
            v1, v2, data_grid = create_2d_grid_data(x1_min=-1.5, x1_max=1.5, x2_min=-1.5, x2_max=1.5, n=n)
            px_est = self.estimate_px_with_IS(data_grid,sd_x=self.params['sd_x'],scale=self.params['scale'],sample_size=self.params['sample_size'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_IS_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
            #px_est = self.estimate_px_with_CF(data_grid,log=False)
            #px_est = px_est.reshape((n,n))
            #plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_CF_density_pre_at_{}.png'.format(self.save_dir,batch_idx))

        elif self.params['dataset'] == 'involute':
            v1, v2, data_grid = create_2d_grid_data(x1_min=-6, x1_max=5, x2_min=-5, x2_max=5, n=n)
            px_est = self.estimate_px_with_IS(data_grid,sd_x=self.params['sd_x'],scale=self.params['scale'],sample_size=self.params['sample_size'],log=False)
            px_est = px_est.reshape((n,n))
            plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_IS_density_pre_at_{}.png'.format(self.save_dir,batch_idx))
            # px_est = self.estimate_px_with_CF(data_grid,log=False)
            # px_est = px_est.reshape((n,n))
            # plot_2d_grid_data(px_est,v1,v2,save_dir='{}/2d_grid_CF_density_pre_at_{}.png'.format(self.save_dir,batch_idx))

    #estimate pdf of x (e.g., p(x)) with importance sampling
    def estimate_px_with_IS(self,x_points, sd_x=0.5, scale=0.5,sample_size=40000,bs=1024,df = 1, log=True):
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

        def px_given_z(zip_list):
            '''
            calculate p(x|z)
            z_points with shape (sample_size, z_dim)
            x_point wish shape (x_dim, )
            '''
            z_points = zip_list[0]
            x_point = zip_list[1]
            #x_points_ = self.g_net(z_points).numpy()
            mean, logvar = self.decode(z_points)
            x_points_ = self.reparameterize(mean, logvar).numpy()
            if log:
                return -self.params['x_dim']*np.log((np.sqrt(2*np.pi)*sd_x))-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2)
            else:
                return 1. / ((np.sqrt(2*np.pi)*sd_x)**self.params['x_dim']) * np.exp(-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2))

        def w_likelihood_ratio(zip_list):
            '''
            calculate w=pz/px
            z_point with shape (z_dim, )
            z_points with shape (sample_size,z_dim)
            '''
            z_point = zip_list[0]
            z_points = zip_list[1]
            if log:
                log_qz = np.sum(t.logpdf(z_point-z_points,df,loc=0,scale=scale),axis=1)
                log_pz = -self.params['z_dim']*np.log(np.sqrt(2*np.pi))-(np.sum((z_points)**2,axis=1))/2.
                return log_pz-log_qz
            else:
                qz =np.prod(t.pdf(z_point-z_points, df,loc=0,scale=scale),axis=1)
                pz = 1. / (np.sqrt(2*np.pi)**self.params['z_dim']) * np.exp(-(np.sum((z_points)**2,axis=1))/2.)
                return pz / qz

        #sample a set of points given each z_point from importance distribution
        def sample_from_qz(z_point):
            '''
            multivariate student t distribution can be constructed from a multivariate Gaussian 
            one can also use t.rvs to sample (see the uncommented line) which is lower
            '''
            S = np.diag(scale**2 * np.ones(self.params['z_dim']))
            z1 = np.random.chisquare(df, sample_size)/df
            z2 = np.random.multivariate_normal(np.zeros(self.params['z_dim']),S,(sample_size,))
            return z_point + z2/np.sqrt(z1)[:,None]
            #return np.hstack([t.rvs(df, loc=value, scale=scale, size=(sample_size,1), random_state=None) for value in z_point])

        z_points_ = self.e_net(x_points).numpy()
        N = len(x_points)
        px_given_z_list=[]
        w_likelihood_ratio_list=[]
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               #ind = tf.range(b*bs, N)
               ind = np.arange(b*bs, N, dtype='int32')
            else:
               #ind = tf.range(b*bs, (b+1)*bs,)
               ind = np.arange(b*bs, (b+1)*bs, dtype='int32')
            batch_x_points = x_points[ind, :]
            batch_z_points_ = z_points_[ind, :]
            batch_z_points_sample_list = program_paral(sample_from_qz,batch_z_points_)
            batch_px_given_z_list = program_paral(px_given_z, zip(batch_z_points_sample_list, batch_x_points))
            batch_w_likelihood_ratio_list = program_paral(w_likelihood_ratio, zip(batch_z_points_, batch_z_points_sample_list))
            px_given_z_list += batch_px_given_z_list
            w_likelihood_ratio_list += batch_w_likelihood_ratio_list

        #calculate p(x)=int(p(x|z)*p(z)dz)=int(p(x|z)*w(z)q(z)dz)=E(p(x|z)*w(z)) where z~q(z)
        if log:
            px_list = list(map(lambda z, x: z+x,px_given_z_list,w_likelihood_ratio_list))
            max_idx_list = [np.where(item==max(item))[0][0] for item in px_list]
            px_est = np.array([np.log(np.sum(np.exp(item[0]-item[0][item[1]])))-np.log(sample_size)+item[0][item[1]] for item in zip(px_list,max_idx_list)])
        else:
            px_list = list(map(lambda z, x: z*x,px_given_z_list,w_likelihood_ratio_list))
            px_est = np.array([np.mean(item) for item in px_list])
        return px_est

    @tf.function
    def get_jacobian(self, data_z):
        """get jacobian matrix.
        Args:
            input: a tensor with shape [batch_size, z_dim].
        Returns:
            returns batch jacobian with shape [batch_size, x_dim, z_dim]
        """
        with tf.GradientTape() as j_tape:
            j_tape.watch(data_z)
            mean, logvar = tf.split(self.g_net(data_z), num_or_size_splits=2, axis=1)
        batch_mean_jacobian = j_tape.batch_jacobian(mean, data_z)
        #batch_logvar_jacobian = j_tape.batch_jacobian(logvar, data_z)
        return batch_mean_jacobian

    def save(self, fname, data):
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()


class RoundtripTV_img(object):
    """ Roundtrip model with 
        1) trainable variance in data space for mnist/cifar10 dataset
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(RoundtripTV_img, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = Covolution2DTransposedNet(input_dim=params['z_dim'], 
                                        model_name='g_net', nb_units=params['g_units'],dataset=params['dataset'] )
        self.e_net = Covolution2DNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = Discriminator_img(input_dim=params['x_dim'],model_name='dx_net',
                                        nb_units=params['dx_units'],dataset=params['dataset'])

        #self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        #self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'])
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'])
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        return {
                "params": self.params,
        }

    @tf.function
    def train_gen_step(self, data_z, data_x):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: date tensor with shape [batch_size, x_dim].
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            x_mean, x_logvar = self.decode(data_z)
            data_x_ = self.reparameterize(x_mean, x_logvar)

            data_z_ = self.e_net(data_x)
            data_z__ = self.e_net(data_x_)

            x_mean_, x_logvar_ = self.decode(data_z_)
            data_x__ = self.reparameterize(x_mean_, x_logvar_, apply_sigmoid=False)

            # variational loss
            # logpx_z = -tf.reduce_mean((data_x - data_x__)**2,axis=1)
            # logqz_x = self.log_normal_pdf(data_z_, z_mean, z_logvar)
            # logpz = self.log_normal_pdf(data_z_, 0., 0.)
            # kl_loss = logqz_x-logpz # here it is not the formula of KL_loss, so will result in negative values(similar)
            # elbo = tf.reduce_mean(logpx_z - kl_loss)

            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=data_x__, labels=data_x))
            #l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = e_loss_adv + g_loss_adv + self.params['alpha']*(l2_loss_z+l2_loss_x)

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))
        return e_loss_adv, g_loss_adv, l2_loss_z, l2_loss_x, g_e_loss

    @tf.function
    def train_disc_step(self, data_z, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: data tensor with shape [batch_size, x_dim].
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            x_mean, x_logvar = self.decode(data_z)
            data_x_ = self.reparameterize(x_mean, x_logvar)
            data_z_ = self.e_net(data_x)
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,v_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss =  dx_loss + dz_loss + self.params['gamma']*(gpx_loss+gpz_loss)

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        return dz_loss, dx_loss, d_loss

    def encode(self, x):
        mean, logvar = tf.split(self.e_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z):
        mean, logvar = self.g_net(z)
        return mean, logvar

    @tf.function
    def sample(self, nb_samples=16,eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(nb_samples, self.params['z_dim']))
        x_mean, x_logvar = self.decode(eps)
        return self.reparameterize(x_mean, x_logvar)

    def reparameterize(self, mean, logvar, apply_sigmoid=True):
        #[32,28,28,1], [32,1]
        eps = tf.random.normal(shape=mean.shape)
        out = tf.transpose(tf.transpose(eps) * tf.exp(tf.transpose(logvar) * .5)) + mean
        if apply_sigmoid:
            return tf.sigmoid(out)
        else:
            return out

    def log_normal_pdf(self, sample, mean, logvar, axis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=axis)

    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500,
            startoff=0, verbose=1, save_format='txt'):
        f_params = open('{}/params.txt'.format(self.save_dir),'w')
        f_params.write(str(self.params))
        #f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data, batch_size=batch_size, normalize=normalize)

        for batch_idx in range(n_iter+1):
            for _ in range(self.params['g_d_freq']):
                batch_x = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(len(batch_x))
                dz_loss, dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            batch_x = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(len(batch_x))
            e_loss_adv, g_loss_adv, l2_loss_z, l2_loss_x, g_e_loss = self.train_gen_step(batch_z, batch_x)

            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : e_loss_adv [%.4f], g_loss_adv [%.4f], l2_loss_z [%.4f],\
                l2_loss_x [%.4f], g_e_loss [%.4f], dz_loss [%.4f], dx_loss [%.4f], d_loss [%.4f]''' \
                %(batch_idx, e_loss_adv, g_loss_adv, l2_loss_z, l2_loss_x, g_e_loss , dz_loss, dx_loss, d_loss)
                if verbose:
                    print(loss_contents)
                f_params.write(loss_contents+'\n')
                #px_est = self.estimate_px_with_CF(self.data_sampler.load_all(),log=True)
                self.evaluate(batch_idx)
                # px_est = self.estimate_px_with_IS(self.data_sampler.load_all(),
                #                                     sd_x=self.params['sd_x'],
                #                                     scale=self.params['scale'],
                #                                     sample_size=self.params['sample_size'])
                # self.save('{}/px_est_{}.{}'.format(self.save_dir, batch_idx, save_format), px_est)

                if self.params['save_model']:
                    ckpt_save_path = self.ckpt_manager.save(batch_idx)
                    print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))
        f_params.close()

    def evaluate(self, batch_idx, n=100):
        import matplotlib.pyplot as plt
        data_x_ = self.sample(nb_samples=16)
        np.save('{}/data_x_at_{}.npy'.format(self.save_dir,batch_idx),data_x_)
        for i in range(data_x_.shape[0]):
            plt.subplot(4, 4, i + 1)
            if self.params['dataset']=='mnist':
                plt.imshow(data_x_[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(data_x_[i, :, :, :])
            plt.axis('off')
        plt.savefig('{}/image_at_batch_{:04d}.png'.format(self.save_dir, batch_idx))
        # for i in range(x_mean.shape[0]):
        #     plt.subplot(4, 4, i + 1)
        #     plt.imshow(x_mean[i, :, :, 0], cmap='gray')
        #     plt.axis('off')
        # plt.savefig('{}/image_mean_at_batch_{:04d}.png'.format(self.save_dir, batch_idx))

    #estimate pdf of x (e.g., p(x)) with importance sampling
    def estimate_px_with_IS(self,x_points, sd_x=0.5, scale=0.5,sample_size=40000,bs=1024,df = 1, log=True):
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

        def px_given_z(zip_list):
            '''
            calculate p(x|z)
            z_points with shape (sample_size, z_dim)
            x_point wish shape (x_dim, )
            '''
            z_points = zip_list[0]
            x_point = zip_list[1]
            #x_points_ = self.g_net(z_points).numpy()
            mean, logvar = self.decode(z_points)
            x_points_ = self.reparameterize(mean, logvar).numpy()
            if log:
                return -self.params['x_dim']*np.log((np.sqrt(2*np.pi)*sd_x))-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2)
            else:
                return 1. / ((np.sqrt(2*np.pi)*sd_x)**self.params['x_dim']) * np.exp(-(np.sum((x_point-x_points_)**2,axis=1))/(2.*sd_x**2))

        def w_likelihood_ratio(zip_list):
            '''
            calculate w=pz/px
            z_point with shape (z_dim, )
            z_points with shape (sample_size,z_dim)
            '''
            z_point = zip_list[0]
            z_points = zip_list[1]
            if log:
                log_qz = np.sum(t.logpdf(z_point-z_points,df,loc=0,scale=scale),axis=1)
                log_pz = -self.params['z_dim']*np.log(np.sqrt(2*np.pi))-(np.sum((z_points)**2,axis=1))/2.
                return log_pz-log_qz
            else:
                qz =np.prod(t.pdf(z_point-z_points, df,loc=0,scale=scale),axis=1)
                pz = 1. / (np.sqrt(2*np.pi)**self.params['z_dim']) * np.exp(-(np.sum((z_points)**2,axis=1))/2.)
                return pz / qz

        #sample a set of points given each z_point from importance distribution
        def sample_from_qz(z_point):
            '''
            multivariate student t distribution can be constructed from a multivariate Gaussian 
            one can also use t.rvs to sample (see the uncommented line) which is lower
            '''
            S = np.diag(scale**2 * np.ones(self.params['z_dim']))
            z1 = np.random.chisquare(df, sample_size)/df
            z2 = np.random.multivariate_normal(np.zeros(self.params['z_dim']),S,(sample_size,))
            return z_point + z2/np.sqrt(z1)[:,None]
            #return np.hstack([t.rvs(df, loc=value, scale=scale, size=(sample_size,1), random_state=None) for value in z_point])

        z_points_ = self.e_net(x_points).numpy()
        N = len(x_points)
        px_given_z_list=[]
        w_likelihood_ratio_list=[]
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               #ind = tf.range(b*bs, N)
               ind = np.arange(b*bs, N, dtype='int32')
            else:
               #ind = tf.range(b*bs, (b+1)*bs,)
               ind = np.arange(b*bs, (b+1)*bs, dtype='int32')
            batch_x_points = x_points[ind, :]
            batch_z_points_ = z_points_[ind, :]
            batch_z_points_sample_list = program_paral(sample_from_qz,batch_z_points_)
            batch_px_given_z_list = program_paral(px_given_z, zip(batch_z_points_sample_list, batch_x_points))
            batch_w_likelihood_ratio_list = program_paral(w_likelihood_ratio, zip(batch_z_points_, batch_z_points_sample_list))
            px_given_z_list += batch_px_given_z_list
            w_likelihood_ratio_list += batch_w_likelihood_ratio_list

        #calculate p(x)=int(p(x|z)*p(z)dz)=int(p(x|z)*w(z)q(z)dz)=E(p(x|z)*w(z)) where z~q(z)
        if log:
            px_list = list(map(lambda z, x: z+x,px_given_z_list,w_likelihood_ratio_list))
            max_idx_list = [np.where(item==max(item))[0][0] for item in px_list]
            px_est = np.array([np.log(np.sum(np.exp(item[0]-item[0][item[1]])))-np.log(sample_size)+item[0][item[1]] for item in zip(px_list,max_idx_list)])
        else:
            px_list = list(map(lambda z, x: z*x,px_given_z_list,w_likelihood_ratio_list))
            px_est = np.array([np.mean(item) for item in px_list])
        return px_est

    @tf.function
    def get_jacobian(self, data_z):
        """get jacobian matrix.
        Args:
            input: a tensor with shape [batch_size, z_dim].
        Returns:
            returns batch jacobian with shape [batch_size, x_dim, z_dim]
        """
        with tf.GradientTape() as j_tape:
            j_tape.watch(data_z)
            mean, logvar = tf.split(self.g_net(data_z), num_or_size_splits=2, axis=1)
        batch_mean_jacobian = j_tape.batch_jacobian(mean, data_z)
        #batch_logvar_jacobian = j_tape.batch_jacobian(logvar, data_z)
        return batch_mean_jacobian

    def save(self, fname, data):
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()