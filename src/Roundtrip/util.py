import numpy as np
import copy
import sys,os
import pandas as pd
from os.path import join
from collections import Counter
from sklearn.preprocessing import StandardScaler
import math
import gzip
try:
    import cPickle as pickle
except:
    import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def Dataset_selector(name):
    if name == 'Semi_acic':
        return Semi_acic_sampler
    elif name=='Semi_ihdp':
        return Semi_ihdp_sampler
    elif name=='Sim_Hirano_Imbens':
        return Sim_Hirano_Imbens_sampler
    elif name=='Sim_Sun':
        return Sim_Sun_sampler
    elif name=='Sim_Colangelo':
        return Sim_Colangelo_sampler
    elif name=='Semi_Twins':
        return Semi_Twins_sampler
    else:
        print('Cannot find the example data sampler: %s!'%name)
        sys.exit()

class Base_sampler(object):
    def __init__(self, x, batch_size, normalize=False, random_seed=123):
        np.random.seed(random_seed)
        self.data_x = np.array(x, dtype='float32')
        self.batch_size = batch_size
        if normalize:
            self.data_x = StandardScaler().fit_transform(self.data_x)
        self.sample_size = len(x)
        self.full_index = np.arange(self.sample_size)
        np.random.shuffle(self.full_index)
        self.idx_gen = self.create_idx_generator(sample_size=self.sample_size)
        
    def create_idx_generator(self, sample_size, random_seed=123):
        while True:
            for step in range(math.ceil(sample_size/self.batch_size)):
                if (step+1)*self.batch_size <= sample_size:
                    yield self.full_index[step*self.batch_size:(step+1)*self.batch_size]
                else:
                    yield np.hstack([self.full_index[step*self.batch_size:],
                                    self.full_index[:((step+1)*self.batch_size-sample_size)]])
                    np.random.shuffle(self.full_index)

    def next_batch(self):
        indx = next(self.idx_gen)
        return self.data_x[indx,:]
    
    def load_all(self):
        return self.data_x


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
    def __init__(self,data_path='datasets/mnist/mnist.pkl.gz'):
        f = gzip.open(data_path, 'rb')
        trn, val, tst = pickle.load(f)
        self.trn_data = trn[0]
        self.trn_label = trn[1]
        self.val_data = val[0]
        self.val_label = val[1]
        
        self.trn_data = np.concatenate([self.trn_data,self.val_data],axis=0)
        self.trn_label = np.concatenate([self.trn_label,self.val_label],axis=0)
        self.trn_one_hot = np.eye(10)[self.trn_label]
        self.trn_data_per_class = [self.trn_data[self.trn_label==i] for i in range(10)]
        self.nb_trn_data_per_class = [len(self.trn_data_per_class[i]) for i in range(10)]
        self.tst_data = tst[0]
        self.tst_label = tst[1]
        self.tst_one_hot = np.eye(10)[self.tst_label]
        self.N = self.trn_data.shape[0]
    def train(self, batch_size, indx = None, label = False):
        if indx is None:
            indx = np.random.randint(low = 0, high = self.N, size = batch_size)
        if label:
            return self.trn_data[indx, :], self.trn_one_hot[indx]
        else:
            return self.trn_data[indx, :]
    def get_batch_by_class(self, batch_size, i):
        assert i in range(10)
        #print(self.nb_trn_data_per_class[i],self.trn_data_per_class[i].shape)
        indx = np.random.randint(low = 0, high = self.nb_trn_data_per_class[i], size = batch_size)
        return self.trn_data_per_class[i][indx,:]
    def load_all(self):
        return self.tst_data, self.tst_label, self.tst_one_hot

class cifar10_sampler(object):
    def __init__(self,data_path='datasets/cifar10'):
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
        #self.val_data = self.trn_data[-int(0.1*(len(self.trn_data))):]
        #self.val_label = self.trn_label[-int(0.1*(len(self.trn_label))):]
        #self.val_one_hot = self.trn_one_hot[-int(0.1*(len(self.trn_one_hot))):]
        #self.N = len(self.trn_data)-len(self.val_data)
        self.N = len(self.trn_data)

        f = open(data_path + '/test_batch', 'rb')
        dict = pickle.load(f)
        tst_data = dict['data']
        tst_data = tst_data.reshape(tst_data.shape[0],3,32,32)
        tst_data = tst_data.transpose(0, 2, 3, 1)
        tst_data = tst_data/256.0
        self.tst_data = tst_data.reshape(tst_data.shape[0],-1)
        self.tst_label = np.array(dict['labels'])
        self.tst_one_hot = np.eye(10)[self.tst_label]
    def train(self, batch_size, indx = None, label = False):
        if indx is None:
            indx = np.random.randint(low = 0, high = self.N, size = batch_size)
        if label:
            return self.trn_data[indx, :], self.trn_one_hot[indx]
        else:
            return self.trn_data[indx, :]
    def load_all(self):
        return self.tst_data, self.tst_label, self.tst_one_hot

# Gaussian mixture sampler
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

    def get_density(self,x_points):
        assert len(x_points.shape)==2
        c = 1./(2*np.pi*self.sigma)
        px = [c*np.mean(np.exp(-np.sum((np.tile(x,[self.total_size,1])-self.X_center.T)**2,axis=1)/(2*self.sigma))) for x in x_points]
        return np.array(px)

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
        self.bound = bound
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
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    
    def get_density(self, data):
        assert data.shape[1]==self.dim
        from scipy.stats import norm
        centers = np.linspace(-self.bound, self.bound, self.n_components)
        prob = []
        for i in range(self.dim):
            p_mat = np.zeros((self.n_components,len(data)))
            for j in range(len(data)):
                for k in range(self.n_components):
                    p_mat[k,j] = norm.pdf(data[j,i], loc=centers[k], scale=self.sd) 
            prob.append(np.mean(p_mat,axis=0))
        prob = np.stack(prob)        
        return np.prod(prob, axis=0)

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
    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean))).astype('float32')

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

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')
    
def create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n=100):
    grid_x1 = np.linspace(x1_min, x1_max, n)
    grid_x2 = np.linspace(x2_min, x2_max, n)
    v1,v2 = np.meshgrid(grid_x1,grid_x2)
    data_grid = np.vstack((v1.ravel(),v2.ravel())).T
    return v1, v2, data_grid

def plot_2d_grid_data(data, v1, v2, save_dir):
    plt.figure()
    plt.rcParams.update({'font.size': 22})
    plt.imshow(data, extent=[v1.min(), v1.max(), v2.min(), v2.max()],
cmap='Blues', alpha=0.9)
    plt.colorbar()
    plt.savefig(save_dir)
    plt.close()

def parse_file(path, sep='\t', header = 0, normalize=False):
    assert os.path.exists(path)
    if path[-3:] == 'npy':
        data = np.load(path).astype('float32')
    elif  path[-3:] == 'csv':
        data = pd.read_csv(path, header=0, sep=sep).values.astype('float32')
    elif path[-3:] == 'txt':
        data = np.loadtxt(path,delimiter=sep).astype('float32')
    else:
        print('File format not recognized, please use .npy, .csv or .txt as input.')
        sys.exit()
    if normalize:
        data = StandardScaler().fit_transform(data)
    return data
