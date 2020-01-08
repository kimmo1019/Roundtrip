import scipy.sparse
import scipy.io
import numpy as np

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


class UniformDataSampler(object):
    def __init__(self, N, n_components, dim, sd):
        self.total_size = N
        self.n_components = n_components
        self.dim = dim 
        self.sd = sd
        np.random.seed(1024)
        self.means = np.random.uniform(-0.5,0.5,(self.n_components,self.dim))
        weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(n_components, size=N, replace=True, p=weights)
        self.X = np.array([np.random.normal(self.means[i],scale=self.sd) for i in self.Y],dtype='float64')



    #for data sampling given batch size
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y
        

class GMMDataSampler(object):    
    def __init__(self):
        self.train_size = 10000
        self.total_size = 10000
        self.test_size = 2000
        self.X_train, self.X_test = self._load_gmm_array()
        self.y_train, self.y_test = self._load_labels()

    def _load_gmm_array(self):
        data_path = 'data/gaus_mix/10_component_gaussian_array.npy'
        data = np.load(data_path)
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        data_train = data[indx[0:self.train_size]]
        data_test = data[indx[-self.test_size:], :]
        return data_train, data_test


    def _load_labels(self):
        data_path =  'data/gaus_mix/10_component_gaussian_label.npy'
        labels = np.load(data_path)
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

class GMM_sampler(object):
    def __init__(self, N, mean, cov, weights=None):
        self.total_size = N
        self.mean = mean
        self.n_components = self.mean.shape[0]
        self.dim = self.mean.shape[1]
        self.cov = cov
        np.random.seed(1024)
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=weights)
        #self.X = np.array([np.random.normal(self.mean[i],scale=0.6) for i in self.Y],dtype='float64')
        self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
        return self.X, self.Y

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

class X_sampler(object):
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
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

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

class Mixture_sampler(object):#sample continuous and discrete lable together
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
    
    def get_batch(self,batch_size):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim))
        label_batch_idx =  np.random.randint(low = 0 , high = self.nb_classes, size = batch_size)
        X_batch_d = np.eye(self.nb_classes)[label_batch_idx]
        return X_batch_c, X_batch_d

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


def sample_labelled_Z(batch, z_dim, sampler = 'one_hot', num_class = 10,  n_cat = 1, label_index = None):

    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((0.10 * np.random.randn(batch, z_dim - num_class*n_cat),
                                       np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])


def reshape_mnist(X):
    return X.reshape(X.shape[0], 28, 28, 1)


def clus_sample_Z(batch, dim_gen=20, dim_c=2, num_class = 10, label_index = None):

    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    batch_mat = np.zeros((batch, num_class* dim_c))
    for b in range(batch):
        batch_mat[b, label_index[b] * dim_c:(label_index[b] + 1) * dim_c] = np.random.normal(loc = 1.0, scale = 0.05, size = (1, dim_c))
    return np.hstack((0.10 * np.random.randn(batch, dim_gen), batch_mat))


def clus_sample_labelled_Z(batch, dim_gen=20, dim_c=2, num_class = 10, label_index = None):
    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    batch_mat = np.zeros((batch, num_class*dim_c))
    for b in range(batch):
        batch_mat[b, label_index[b] * dim_c:(label_index[b] + 1) * dim_c] = np.random.normal(loc=1.0, scale = 0.05, size = (1, dim_c))
    return label_index, np.hstack((0.10 * np.random.randn(batch, dim_gen), batch_mat))



def sample_info(batch, z_dim, sampler = 'one_hot', num_class = 10,  n_cat = 1, label_index = None):
    if sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack(
            (np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((np.random.randn(batch, z_dim - num_class*n_cat),
                                       np.tile(np.eye(num_class)[label_index], (1, n_cat))))


if __name__=='__main__':

    l = sample_Z(10, 22, 'mul_cat', 10, 2)
    print(l)

