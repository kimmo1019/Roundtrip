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
import pandas as pd
from os.path import join
from collections import Counter
import cPickle as pickle
import gzip

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

#outliner dataset (http://odds.cs.stonybrook.edu/)
class Outlier_sampler(object):
    def __init__(self,data_path='datasets/OOD/Shuttle/data.npz'):
        data_dic = np.load(data_path)
        self.X_train, self.X_val,self.X_test,self.label_test = self.normalize(data_dic)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data_dic):
        data = data_dic['arr_0']
        label = data_dic['arr_1']
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        label_test = label[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test, label_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None


#UCI dataset
class UCI_sampler(object):
    def __init__(self,data_path='datasets/AReM/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

#miniboone dataset
class miniboone_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/miniboone/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None
#power dataset
class power_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/power/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        N = data.shape[0]
        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        voltage_noise = 0.01*rng.rand(N, 1)
        gap_noise = 0.001*rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu)/s
        data_validate = (data_validate - mu)/s
        data_test = (data_test - mu)/s
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

#power dataset
class gas_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/gas/ethylene_CO.pickle'):
        data = pd.read_pickle(data_path)
        self.X_train, self.X_val,self.X_test = self.normalize(data)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        C = data.corr()
        A = C > 0.98
        B = A.as_matrix().sum(axis=1)
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            C = data.corr()
            A = C > 0.98
            B = A.as_matrix().sum(axis=1)
        data = (data-data.mean())/data.std()
        data = data.as_matrix()
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1*data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
        return data_train, data_validate, data_test

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

#HEPMASS dataset
class hepmass_sampler(object):
    def __init__(self,data_path='datasets/HEPMASS/'):
        self.X_train, self.X_val,self.X_test = self.normalize(data_path)
        self.Y=None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data_path):
        data_train = pd.read_csv(filepath_or_buffer=join(data_path, "1000_train.csv"), index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=join(data_path, "1000_test.csv"), index_col=False)
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu)/s
        data_test = (data_test - mu)/s
        data_train, data_test = data_train.as_matrix(), data_test.as_matrix()
        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        N = data_train.shape[0]
        N_validate = int(N*0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

class mnist_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/mnist/mnist.pkl.gz'):
        f = gzip.open(data_path, 'rb')
        trn, val, tst = pickle.load(f)
        self.trn_data = trn[0]
        self.trn_label = trn[1]
        self.trn_one_hot = np.eye(10)[self.trn_label]
        self.tst_data = tst[0]
        self.tst_label = tst[1]
        self.N = self.trn_data.shape[0]
        self.mean = 0
        self.sd = 0
    def train(self, batch_size, indx = None, label = False):
        if indx is None:
            indx = np.random.randint(low = 0, high = self.N, size = batch_size)
        if label:
            return self.trn_data[indx, :], self.trn_one_hot[indx]
        else:
            return self.trn_data[indx, :]
    def load_all(self):
        return self.trn_data, self.trn_label, self.trn_one_hot

class cifar10_sampler(object):
    def __init__(self,data_path='/home/liuqiao/software/maf/data/cifar10'):
        trn_data = []
        trn_label = []
        for i in xrange(1, 6):
            f = open(data_path + '/data_batch_' + str(i), 'rb')
            dict = pickle.load(f)
            trn_data.append(dict['data'])
            trn_label.append(dict['labels'])
            f.close()
        trn_data = np.concatenate(trn_data, axis=0)
        trn_data = trn_data.reshape(trn_data.shape[0],3,32,32)
        trn_data = trn_data.transpose(0, 2, 3, 1)
        trn_data = trn_data/256.0
        self.trn_data = trn_data.reshape(trn_data.shape[0],-1)
        self.trn_label = np.concatenate(trn_label, axis=0)
        self.trn_one_hot = np.eye(10)[self.trn_label]
        self.N = self.trn_data.shape[0]
        f = open(data_path + '/test_batch', 'rb')
        dict = pickle.load(f)
        tst_data = dict['data']
        tst_data = tst_data.reshape(tst_data.shape[0],3,32,32)
        tst_data = tst_data.transpose(0, 2, 3, 1)
        tst_data = tst_data/256.0
        self.tst_data = tst_data.reshape(tst_data.shape[0],-1)
        self.tst_label = np.array(dict['labels'])
        self.mean = 0
        self.sd = 0
    def train(self, batch_size, indx = None, label = False):
        if indx is None:
            indx = np.random.randint(low = 0, high = self.N, size = batch_size)
        if label:
            return self.trn_data[indx, :], self.trn_one_hot[indx]
        else:
            return self.trn_data[indx, :]
    def load_all(self):
        return self.trn_data, self.trn_label, self.trn_one_hot

# Gaussian mixture sampler by either given parameters or random component centers and fixed sd
class GMM_sampler(object):
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            #self.mean = np.random.uniform(-0.5,0.5,(self.n_components,self.dim))
            self.mean = np.random.uniform(-5,5,(self.n_components,self.dim))
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
        self.X_train, self.X_val,self.X_test = self.split(self.X)

    def split(self,data):
        #N_test = int(0.1*data.shape[0])
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]
        #N_validate = int(0.1*data.shape[0])
        N_validate = 2000
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = len(self.X_train), size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y

#Swiss roll (r*sin(scale*r),r*cos(scale*r)) + Gaussian noise
class Swiss_roll_sampler(object):
    def __init__(self, N, theta=2*np.pi, scale=2, sigma=0.4):
        np.random.seed(1024)
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        params = np.linspace(0,self.theta,self.total_size)
        self.X_center = np.vstack((params*np.sin(scale*params),params*np.cos(scale*params)))
        self.X = self.X_center.T + np.random.normal(0,sigma,size=(self.total_size,2))
        np.random.shuffle(self.X)
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.Y = None
        self.mean = 0
        self.sd = 0

    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y

#Gaussian mixture + normal + uniform distribution
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

#each dim is a gmm
class GMM_indep_sampler(object):
    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
    def generate_gmm(self,weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i],self.sd) for i in Y],dtype='float64')
    def split(self,data):
        #N_test = int(0.1*data.shape[0])
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]
        #N_validate = int(0.1*data.shape[0])
        N_validate = 2000
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y


#Gaussian + uniform distribution
class Multi_dis_sampler(object):
    def __init__(self, N, dim):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        assert dim >= 5
        #first two dims are GMM
        self.mean = np.array([[0.2,0.3],[0.7,0.8]])
        self.cov = [0.1**2*np.eye(2),0.1**2*np.eye(2)]
        comp_idx = np.random.choice(2, size=self.total_size, replace=True)
        self.X_gmm = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in comp_idx],dtype='float64')
        #dim 3 is a normal
        self.X_gau = np.random.normal(0.5, 0.1, size=(self.total_size,1))
        #dim 4 is a uniform
        self.X_uni = np.random.uniform(0,1,size=(self.total_size,1))
        #dim >=5 is a GMM for each dim
        self.centers=np.array([0.2,0.6])
        self.sd = np.array([0.1,0.05])
        self.X_indep_gmm = np.vstack([self.generate_gmm(self.centers,self.sd) for _ in range(dim-4)]).T
        self.X = np.hstack((self.X_gmm,self.X_gau,self.X_uni,self.X_indep_gmm))
        self.X_train, self.X_val,self.X_test = self.split(self.X)
    def generate_gmm(self,centers,sd):
            Y = np.random.choice(2, size=self.total_size, replace=True)
            return np.array([np.random.normal(centers[i],sd[i]) for i in Y],dtype='float64')
        
    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = len(self.X_train), size = batch_size)
        return self.X[indx, :]
    def get_single_density(self,data):
        #gmm
        p1 = 1./(np.sqrt(2*np.pi)*0.1) * np.exp(-np.sum((self.mean[0]-data[:2])**2) / (2*0.1**2)) 
        p2 = 1./(np.sqrt(2*np.pi)*0.1) * np.exp(-np.sum((self.mean[1]-data[:2])**2) / (2*0.1**2)) 
        p_gmm = (p1+p2)/2.
        #Gaussian
        p_gau = 1./(np.sqrt(2*np.pi)*0.1)**2 * np.exp(-np.sum((0.5-data[2])**2) / (2*0.1**2)) 
        #Uniform
        p_uni = 1
        #indep gmm
        p_indep_gmm = 1
        for i in range(4,self.dim):
            p1 = 1./(np.sqrt(2*np.pi)*self.sd[0]) * np.exp(-np.sum((self.centers[0]-data[i])**2) / (2*self.sd[0]**2)) 
            p2 = 1./(np.sqrt(2*np.pi)*self.sd[1]) * np.exp(-np.sum((self.centers[1]-data[i])**2) / (2*self.sd[1]**2)) 
            p_indep_gmm *= (p1+p2)/2.
        return np.prod([p_gmm,p_gau,p_uni,p_indep_gmm])
    def get_all_density(self,batch_data):
        assert batch_data.shape[1]==self.dim
        p_all = map(self.get_single_density,batch_data)
        return np.array(p_all)


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
    def __init__(self, nb_classes, N, dim, sampler='normal',scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, 1, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes)[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X_c[indx, :],self.X_d[indx, :],indx
    
    def get_batch(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim))
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes)[label_batch_idx]
        return X_batch_c,X_batch_d,label_batch_idx

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
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data


if __name__=='__main__':
    ys = UCI_sampler('datasets/YearPredictionMSD/data.npy')
    print ys.X_train.shape, ys.X_val.shape,ys.X_test.shape