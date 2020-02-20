from __future__ import division
import scipy.sparse
import scipy.io
import numpy as np
import copy
from scipy.special import polygamma
import scipy.special
from scipy.stats import t, uniform, norm, truncnorm, invgamma, gamma
from scipy import pi
from tqdm import tqdm
import sys

#pbmc ~68k single cell RNA-seq data
class DataSampler(object):
    def __init__(self):
        self.train_size = 68260
        self.total_size = 68260
        self.test_size = 2000
        self.X_train, self.X_test = self._load_gene_mtx()
        self.y_train, self.y_test = self._load_labels()
    def _read_mtx(self, filename):
        buf = scipy.io.mmread(filename)
        return buf

    def _load_gene_mtx(self):
        data_path = 'data/pbmc68k/filtered_mat.txt'
        data = np.loadtxt(data_path,delimiter=' ',skiprows=1,usecols=range(1,68261))
        data = data.T
        scale = np.max(data)
        data = data / scale
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        data_train = data[indx[0:self.train_size], :]
        data_test = data[indx[-self.test_size:], :]

        return data_train, data_test

    def _load_labels(self):
        data_path = 'data/pbmc68k/label_info.txt'
        labels = np.array([int(item.split('\t')[-1].strip()) for item in open(data_path).readlines()])
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        labels_train = labels[indx[0:self.train_size]]
        labels_test = labels[indx[-self.test_size:]]
        return labels_train, labels_test

    # for data sampling given batch size
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.train_size, size = batch_size)

        if label:
            return self.X_train[indx, :], self.y_train[indx].flatten()
        else:
            return self.X_train[indx, :]

    def validation(self):
        return self.X_train[-5000:,:], self.y_train[-5000:].flatten()
    
    def training(self):
        return self.X_train, self.y_train
    
    def test(self):
        return self.X_test, self.y_test

    def load_all(self):
         return np.concatenate((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test))
        
# Gaussian mixture sampler by either given parameters or random component centers and fixed sd
class GMM_sampler(object):
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            self.mean = np.random.uniform(-0.5,0.5,(self.n_components,self.dim))
        else:
            assert cov is not None    
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=self.weights)
        if mean is None:
            self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X[indx, :], self.Y[indx]
        else:
            return self.X[indx, :]
    def resample(self,nb_sample):
        Y = np.random.choice(self.n_components, size=nb_sample, replace=True, p=self.weights)
        try:
            X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in Y],dtype='float64')
        except AttributeError:
            X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in Y],dtype='float64')
        return X,Y
    def load_all(self):
        return self.X, self.Y

#Swiss roll (r*sin(scale*r),r*cos(scale*r)) + Gaussian noise
class Swiss_roll_sampler(object):
    def __init__(self, N, theta=2*np.pi, scale=2, sigma=0.2):
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        np.random.seed(1024)
        params = np.linspace(0,self.theta,self.total_size)
        self.X_center = np.vstack((params*np.sin(scale*params),params*np.cos(scale*params)))
        self.X = self.X_center.T + np.random.normal(0,sigma,size=(self.total_size,2))
        self.Y = None
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y

#Gaussian mixture + uniform distribution
class GMM_Uni_sampler(object):
    def __init__(self, N, mean, cov, norm_dim=2,uni_dim=10,weights=None):
        self.total_size = N
        self.mean = mean
        self.n_components = self.mean.shape[0]
        self.norm_dim = norm_dim
        self.uni_dim = uni_dim
        self.cov = cov
        np.random.seed(1024)
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        #self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        self.X_gmm = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_normal = np.random.normal(0.5, np.sqrt(0.1), (self.total_size,self.norm_dim))
        self.X_uni = np.random.uniform(-0.5,0.5,(self.total_size,self.uni_dim))
        self.X = np.concatenate([self.X_gmm,self.X_normal,self.X_uni],axis = 1)
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y

class Uniform_sampler(object):
    def __init__(self, N, dim, mean):
        self.total_size = N
        self.dim = dim 
        self.mean = mean
        np.random.seed(1024)
        self.centers = np.random.uniform(-0.5,0.5,(self.dim,))
        #print self.centers
        #self.X = np.random.uniform(self.centers-0.5,self.centers+0.5,size=(self.total_size,self.dim))
        self.Y = None
        self.X = np.random.uniform(self.mean-0.5,self.mean+0.5,(self.total_size,self.dim))

    def get_batch(self, batch_size):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,self.dim))
    #for data sampling given batch size
    def train(self, batch_size, label = False):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,self.dim))

    def load_all(self):
        return self.X, self.Y

class Gaussian_sampler(object):
    def __init__(self, N, mean, sd=1):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    
    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

    def load_all(self):
        return self.X, self.Y

#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sampler='normal',scale=0.1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes)[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X_c[indx, :], self.X_d[indx, :]
    
    def get_batch(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes)[label_batch_idx]
        return X_batch_c, X_batch_d

    def load_all(self):
        return self.X_c, self.X_d, self.label_idx

#sample continuous (Gaussian Mixture) and discrete (Catagory) latent variables together
class Mixture_sampler_v2(object):
    def __init__(self, nb_classes, N, dim, weights=None,sd=0.5):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        np.random.seed(1024)
        if nb_classes<=dim:
            self.mean = np.zeros((nb_classes,dim))
            self.mean[:,:nb_classes] = np.eye(nb_classes)
        else:
            if dim==2:
                self.mean = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
            else:
                self.mean = np.zeros((nb_classes,dim))
                self.mean[:,:2] = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
        self.cov = [sd**2*np.eye(dim) for item in range(nb_classes)]
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        self.Y = np.random.choice(self.nb_classes, size=N, replace=True, p=weights)
        self.X_c = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_d = np.eye(self.nb_classes)[self.Y]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X_c[indx, :], self.X_d[indx, :], self.Y[indx, :]
        else:
            return self.X_c[indx, :], self.X_d[indx, :]

    def get_batch(self,batch_size,weights=None):
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        return self.X_c[label_batch_idx, :], self.X_d[label_batch_idx, :]
    def predict_onepoint(self,array):#return component index with max likelyhood
        from scipy.stats import multivariate_normal
        assert len(array) == self.dim
        return np.argmax([multivariate_normal.pdf(array,self.mean[idx],self.cov[idx]) for idx in range(self.nb_classes)])

    def predict_multipoints(self,arrays):
        assert arrays.shape[-1] == self.dim
        return map(self.predict_onepoint,arrays)
    def load_all(self):
        return self.X_c, self.X_d, self.label_idx


def sample_Z(batch, z_dim , sampler = 'one_hot', num_class = 10, n_cat = 1, label_index = None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class*n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        #return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))
        return np.hstack((0.10 * np.random.normal(0,1,(batch, z_dim-num_class)), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15*np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.pool[idx])[0]
            self.pool[idx][0] = data[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.pool[idx])[1]
            self.pool[idx][1] = data[1]
            return [tmp1, tmp2]
        else:
            return data

class Bayesian_sampler(object):
    def __init__(self, N, dim1=10, dim2=5+1):
        self.total_size = N
        self.dim1 = dim1#y
        self.dim2 = dim2#theta
        np.random.seed(1024)
        self.data = np.load('TS-data_block1.npy')
        
        self.X1 = self.data[:N,-self.dim1:]
        self.X2 = self.data[:N,:self.dim2]
        assert self.X2.shape[1]==self.dim2
        assert self.X1.shape[1]==self.dim1
        self.X = np.hstack((self.X1,self.X2))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]
    
    def get_batch(self,batch_size,weights=None):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]

    def load_all(self):
        return self.X1,self.X2


class SV_sampler(object):#stochastic volatility model
    def __init__(self, theta_init, sample_size, block_size=10,seed = 1):
        np.random.seed(seed)
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        self.sample_size = sample_size
        self.theta_init = theta_init
        self.block_size = block_size
        #self.y_true, _, self.h_true = self.generate_data(sample_size=1,time_step=1000,use_prior=True)
        #print self.y_true[0,-5:]

    def generate_data(self,sample_size,time_step,theta=None,h_0=None,use_prior=False, seed = 1):
        np.random.seed(seed)
        assert len(self.theta_init)==5
        h_t = np.empty((sample_size, time_step), dtype=np.float64)#log-volatility
        y_t = np.empty((sample_size, time_step), dtype=np.float64)#observation data
        if use_prior:
            theta = np.empty((sample_size, len(self.theta_init)), dtype=np.float64)
            #theta[:,0] = self.generate_mu(sample_size)
            theta[:,0] = np.random.normal(loc=0.0314,scale=1.0,size=(sample_size,))
            theta[:,1] = self.generate_phi(sample_size)
            theta[:,2] = self.generate_sigma2(sample_size)
            #theta[:,3] = self.generate_nu(sample_size)
            #theta[:,4] = self.generate_lambda(sample_size)
            #theta[0,:] = self.theta_init
            theta[0,:] = [0.0314, 0.9967, 0.0107, 19.6797, -1.1528]
        mu = theta[:, 0]
        phi = theta[:, 1]
        sigma2 = theta[:, 2]
        sigma = sigma2 ** 0.5
        #nu = theta[:, 3]
        #lambda_ = theta[:, 4]

        if use_prior:
            #h_t[:, 0] = norm.rvs(size=sample_size) * (sigma2 / (1 - phi ** 2)) ** 0.5 + mu
            h_t[:, 0] = 0
        else:
            h_t[:, 0] = mu + phi * (h_0 - mu) + sigma * norm.rvs(size=sample_size)

        for t_ in range(1, time_step):
            h_t[:, t_] = mu + phi * (h_t[:, t_-1] - mu) + sigma * norm.rvs(size=sample_size)
           
        #generate y_t
        for i in range(sample_size):
            #zeta, omega = self.get_zeta_omega(lambda_=lambda_[i], nu=nu[i])
            #epsilon = self.generate_skew_student_t(sample_size=time_step, zeta=zeta, omega=omega, lambda_=lambda_[i], nu=nu[i])
            epsilon = self.generate_normal(time_step)
            y_t[i, :] = np.exp(h_t[i, :] / 2) * epsilon
        return y_t, theta[:,:3], h_t
   
    def generate_normal(self,sample_size,low=0.,high=1.):
        return np.random.normal(size=sample_size)

    def generate_mu(self,sample_size):
        return norm.rvs(scale=1, size=sample_size)
    
    def generate_phi(self,sample_size):
        my_mean = 0.95
        my_std = 10
        myclip_a = -1
        myclip_b = 1
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        return truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=sample_size)

    def generate_sigma2(self,sample_size):
        return invgamma.rvs(a=2.5, scale=0.025, size=sample_size)

    def prior_of_nu(self, nu):
        first = nu / (nu + 3)
        first **= 0.5
        second = polygamma(1, nu / 2) - polygamma(1, (nu + 1) / 2) - 2 * (nu + 3) / nu / (nu + 1) ** 2
        second **= 0.5
        return first * second

    def generate_nu(self, sample_size, left = 10, right = 40):
        out = []
        temp = self.prior_of_nu(left)
        while len(out) < sample_size:
            nu = uniform.rvs() * (right - left) + left
            if uniform.rvs() < self.prior_of_nu(nu) / temp:
                out.append(nu)
        return np.array(out, dtype=np.float64)

    def generate_lambda(self, sample_size):
        # return t.rvs(df=0.5, loc=0.0, scale=pi ** 2 / 4, size=sample_size)
        return norm.rvs(loc=0.0, scale=pi ** 2 / 4, size=sample_size)

    def get_zeta_omega(self,lambda_, nu):
        k1 = (nu / 2) ** 0.5 * scipy.special.gamma(nu / 2 - 0.5) / scipy.special.gamma(nu / 2)
        k2 = nu / (nu - 2)
        delta = lambda_ / (1 + lambda_ ** 2) ** 0.5
        omega = (k2 - 2 / pi * k1 ** 2 * delta ** 2) ** (-0.5)
        zeta = - (2 / pi) ** 0.5 * k1 * omega * delta
        return zeta, omega

    def generate_skew_student_t(self, sample_size, zeta, omega, lambda_, nu):
        delta = lambda_ / (1 + lambda_ ** 2) ** 0.5
        w = truncnorm.rvs(a=0, b=float('inf'), size=sample_size)
        epsilon = norm.rvs(size=sample_size)
        u = gamma.rvs(a=nu / 2, scale=2 / nu, size=sample_size)
        return zeta + u ** (-0.5) * omega * (delta * w + (1 - delta ** 2) ** 0.5 * epsilon)

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]
    
    def get_batch(self,batch_size,weights=None):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X1[indx, :], self.X2[indx, :]

    def load_all(self):
        return self.X1,self.X2

# model: y_t = A cos ( 2 pi omega t + phi ) + sigma w_t, w_t ~ N (0, 1)
# T = 1/omega
# parameters: (omega, phi, logsigma, logA)
# prior:
#     logA ~ N (0, 1)
#     phi ~ Unif(0, pi)
#     omega ~ Unif(0, 0.1)
#     logsigma ~ N (0, 1)
class Cosine_sampler(object):
    def __init__(self, omega=1./10,sigma=0.1,iteration=10,block_size=10):
        self.omega = omega
        self.sigma = sigma
        self.block_size = block_size
        self.observation = np.zeros(iteration*block_size)
    def generate_data(self,sample_size,time_step,seed=0):#time series data t=1,2,3...
        np.random.seed(seed)
        theta = np.empty((sample_size, 4), dtype=np.float64)
        data = np.empty((sample_size, time_step), dtype=np.float64)
        theta[:, 0] = np.random.normal(size=sample_size)
        theta[:, 1] = np.random.uniform(low=0, high=np.pi, size=sample_size)
        theta[:, 2] = 1./80
        theta[:, 3] = -10
        #theta[:, 0] = np.random.uniform(low=0, high=0.1, size=sample_size)
        #theta[:, 1] = np.random.uniform(low=0, high=2 * np.pi, size=sample_size)
        #theta[:, 2] = np.random.normal(size=sample_size)
        #theta[:, 2] = np.random.uniform(low=-5, high=-2, size=sample_size)
        #theta[:, 3] = np.random.normal(size=sample_size)
        #theta[0, :] = self.theta_init
        theta[0,:2] = [np.log(2),np.pi / 4]
        #theta[0, :] = [1 / 80, np.pi / 4, 0, np.log(2)]

        logA, phi, omega, logsigma = theta.transpose()
        sigma = np.exp(logsigma)
        A = np.exp(logA)

        for t in range(time_step):
            data[:, t] = A * np.cos(2 * np.pi * omega * (t + 1) + phi) + sigma * np.random.normal(size=sample_size)
        return data, theta[:,:2]
    def generate_data2(self,sample_size,time_step=10,seed=0):#time series data t is continous from [0,1]
        np.random.seed(seed)
        theta = np.empty((sample_size, 4), dtype=np.float64)
        data = np.empty((sample_size, 2), dtype=np.float64)
        theta[:, 0] = np.random.normal(size=sample_size)
        theta[:, 1] = np.random.uniform(low=0, high=np.pi, size=sample_size)
        theta[:, 2] = 0.5
        theta[:, 3] = -10
        #theta[:, 0] = np.random.uniform(low=0, high=0.1, size=sample_size)
        #theta[:, 1] = np.random.uniform(low=0, high=2 * np.pi, size=sample_size)
        #theta[:, 2] = np.random.normal(size=sample_size)
        #theta[:, 2] = np.random.uniform(low=-5, high=-2, size=sample_size)
        #theta[:, 3] = np.random.normal(size=sample_size)
        #theta[0, :] = self.theta_init
        #theta[0,:2] = [np.log(2),np.pi / 4]
        #theta[0,:2] = [0.0, np.pi / 2]
        theta[0,:2] = [1, 3.*np.pi / 4]
        #theta[0, :] = [1 / 80, np.pi / 4, 0, np.log(2)]

        logA, phi, omega, logsigma = theta.transpose()
        sigma = np.exp(logsigma)
        A = np.exp(logA)
        for i in range(sample_size):
            if i<time_step:
                t = float(i) / time_step
                y_t = A[0] * np.cos(2 * np.pi * omega[0] * t + phi[0]) + sigma[0] * np.random.normal()
            else:
                t = np.random.uniform()
                y_t = A[i] * np.cos(2 * np.pi * omega[i] * t + phi[i]) + sigma[i] * np.random.normal()
            data[i,:] = [y_t,t]
        return data, theta[:,:2] 
        
    def generate_data3(self,sample_size,iteration,prior=None,seed=0):#minibatch=1 in the above generate_data2()
        #np.random.seed(seed)
        np.random.seed(iteration)
        params = np.empty((sample_size, 2), dtype=np.float64)
        data = np.empty((sample_size, self.block_size), dtype=np.float64)
        if prior is None:
            params[:, 0] = np.random.normal(size=sample_size)
            params[:, 1] = np.random.uniform(low=-np.pi, high=np.pi, size=sample_size)
        else:
            params[:,:2] = prior
        #params[0,:2] = [np.log(2),np.pi / 4]# (0.69, 0.78)
        params[0,:] = [0.0, np.pi / 2]  # (0, 1.57)
        #params[0,:2] = [1, 3.*np.pi / 4]  #(1, 2.35)
        #params[0, :] = [1 / 80, np.pi / 4, 0, np.log(2)] 
        logA, phi = params.transpose()
        A = np.exp(logA)
    
        #t = np.linspace(self.block_size*iteration,self.block_size*(iteration+1)-1,self.block_size)
        t = np.random.uniform(self.block_size*iteration,self.block_size*(iteration+1),size=self.block_size)
        #Note that the first row is the observation data
        for i in range(sample_size):
            data[i,:] = A[i] * np.cos(2 * np.pi * self.omega * t + phi[i]) + self.sigma * np.random.normal(size=self.block_size)
        return data, params, t

    def get_posterior(self, data, t, res = 100):
        #get the truth params
        _,params,_ = self.generate_data3(1,0,self.block_size)
        logA, phi = params[0,:]

        #calculate posterior using bayesian formula
        def cal_bayesian_posterior(data,t,params_list,use_log=True):
            logA, phi = params_list
            A = np.exp(logA)
            log_params_prior = -logA**2/2
            log_likelyhood = np.array([-np.sum((data-A_*np.cos(2*np.pi * self.omega *t+phi_))**2) / (2*self.sigma**2) \
                for A_, phi_ in zip(A, phi)])
            if use_log:
                return log_params_prior+log_likelyhood
            else:
                return np.exp(log_params_prior+log_likelyhood)

        #sample theta by Metroplis-Hasting algorithm 
        def sample_posterior(data, t, sample_size=1000, chain_len=500, seed=0):
            np.random.seed(seed)
            para_temp = np.zeros((2, sample_size), dtype=np.float64)
            #starting states of Markov Chain
            para_temp[0, :] = 0 # logA
            para_temp[1, :] = 0  #phi

            for _ in tqdm(range(chain_len)):
                para_propose = para_temp + np.random.normal(scale=0.1, size=(2, sample_size))
                #para_propose[0, :] %= 0.1
                para_propose[1, :] += 2*np.pi
                para_propose[1, :] %= (4 * np.pi)
                para_propose[1, :] -= 2*np.pi
                #para_propose[1, :] = para_propose[1, :]%(2 * np.pi)-np.pi 
                mask = cal_bayesian_posterior(data, t, para_propose) > np.log(np.random.uniform(size=sample_size)) \
                    + cal_bayesian_posterior(data, t, para_temp)
                para_temp[:, mask] = para_propose[:, mask]
            return para_temp.T

        logA_axis = np.linspace(logA-2,logA+2,res)
        phi_axis = np.linspace(phi-3*np.pi,phi+np.pi,res)
        X,Y = np.meshgrid(logA_axis,phi_axis)
        params_stacks = np.vstack([X.ravel(), Y.ravel()]) #shape (2, N*N)
        #log_posterior_list = map(cal_beyesian_posterior,theta_list)
        bayesian_posterior_stacks = cal_bayesian_posterior(data, t, params_stacks)
        bayesian_posterior_mat = np.reshape(bayesian_posterior_stacks,(len(phi_axis),len(logA_axis)))
        params_sampled = sample_posterior(data,t)
        return bayesian_posterior_mat, logA_axis, phi_axis, params_sampled



if __name__=='__main__':
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    from  sklearn.mixture import GaussianMixture
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    s = Cosine_sampler(block_size=10)
    data_list,t_list=[],[]
    for i in range(10):
        print len(data_list),len(t_list)
        if i==0:
            data, theta, t = s.generate_data3(100000,i)
        else:
            data, theta, t = s.generate_data3(100000,i,prior=theta)
        data_list.append(data[0,:])
        t_list.append(t)
        log_prob,axis_x,axis_y,sampled_theta = s.get_posterior(np.concatenate(data_list,axis=0),np.concatenate(t_list,axis=0))
        fig, ax = plt.subplots(1, 2)
        ax[0].hist(sampled_theta[:,0], bins=30,alpha=0.75)
        ax[1].hist(sampled_theta[:,1], bins=30,alpha=0.75)
        plt.savefig('data/bayes_infer/posterior2/iter_%d_sampled_posterior.png'%i)
        plt.close('all')
        prob = np.exp(log_prob)
        z_min, z_max = -np.abs(prob).max(), np.abs(prob).max()
        plt.imshow(prob, cmap='RdBu', vmin=z_min, vmax=z_max,
                extent=[axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()],
                interpolation='nearest', origin='lower')
        plt.title('Beyesian posterior')
        plt.colorbar()
        plt.savefig('data/bayes_infer/posterior2/iter_%d_posterior_2d.png'%i)
        plt.close('all')
        
    sys.exit()
    data, theta, t = s.generate_data3(50000,0)
    print data[0],t
    data, theta, t = s.generate_data3(50000,1,prior=np.zeros(theta.shape))
    print data[0],t
    data, theta, t = s.generate_data3(50000,2,prior=np.zeros(theta.shape))
    print data[0],t
    sys.exit(0)
    log_prob,axis_x,axis_y,sampled_theta = s.get_posterior(data[0,:],t,0)
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(sampled_theta[:,0], bins=100,alpha=0.75)
    ax[1].hist(sampled_theta[:,1], bins=100,alpha=0.75)
    plt.savefig('sampled_posterior.png')
    plt.close()
    print log_prob.shape
    print np.max(log_prob)
    print(np.where(log_prob==np.max(log_prob)))
    # sns.set(style='whitegrid', color_codes=True)
    # sns.heatmap(np.exp(prob))
    prob = np.exp(log_prob)
    z_min, z_max = -np.abs(prob).max(), np.abs(prob).max()
    plt.imshow(prob, cmap='RdBu', vmin=z_min, vmax=z_max,
            extent=[axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()],
            interpolation='nearest', origin='lower')
    plt.title('posterior')
    plt.colorbar()
    plt.savefig('posterior_map.png')
    sys.exit()
    print data.shape
    print theta[0]
    plt.plot(data[0,:50])
    plt.xlabel('t')
    plt.ylabel('y_t')
    plt.savefig('a.png')
    sys.exit()
    
    X, Y = np.mgrid[-2:2:100j, -2:2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    print positions.shape

    kernel = stats.gaussian_kde(values)
    Z = kernel(positions)
    #Z = np.reshape(kernel(positions).T, X.shape)
    print Z.shape
    Z2 = kernel.pdf(positions)
    print Z[0:4],Z2[0:4]
    X = np.random.normal(size=(100,3))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    log_density = kde.score_samples(X)
    log_density1 = kde.score(X)
    print len(log_density)
    print log_density[:3],sum(log_density)
    print log_density1
    a=np.log(2)
    print np.e**a
    sys.exit()
    a=np.ones((3,5))
    c=np.ones((2,5))
    b.append(a)
    b.append(c)
    print np.concatenate(b,axis=0)
    sys.exit()
    ys = Gaussian_sampler(N=10000,mean=np.zeros(5),sd=1.0)
    print ys.get_batch(3)
    print ys.get_batch(3)
    print ys.get_batch(3)
    a=np.array([0.0314, 0.9967, 0.0107, 19.6797, -1.1528])
    a=np.ones((4,3))
    import random
    print random.sample(a,2)
    sys.exit()
    ys = SV_sampler(np.array([0.0314, 0.9967, 0.0107, 19.6797, -1.1528]),1)


