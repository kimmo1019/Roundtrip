import numpy as np
import sys, os
import math
import argparse
import importlib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import util
from scipy.stats import rankdata
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

def create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n=100):
    grid_x1 = np.linspace(x1_min, x1_max, n)
    grid_x2 = np.linspace(x2_min, x2_max, n)
    v1,v2 = np.meshgrid(grid_x1,grid_x2)
    data_grid = np.vstack((v1.ravel(),v2.ravel())).T
    return v1, v2, data_grid


def visualization_2d(x1_min, x1_max, x2_min, x2_max, sd_y, scale, n=100):
    v1, v2, data_grid = create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n)
    py = RTM.estimate_py_with_IS(data_grid,epoch,sd_y=sd_y,scale=scale,sample_size=40000,log=False,save=False)
    py = py.reshape((n,n))
    plt.figure()
    plt.rcParams.update({'font.size': 22})
    plt.imshow(py, extent=[v1.min(), v1.max(), v2.min(), v2.max()],
cmap='Blues', alpha=0.9)
    plt.colorbar()
    plt.savefig('%s/2d_grid_density_pre.png'%path.rstrip('/'))
    plt.close()

def odd_evaluate():
    X_test = RTM.y_sampler.X_test
    X_train = RTM.y_sampler.X_train
    label_test = RTM.y_sampler.label_test
    #one-class SVM
    clf = OneClassSVM(gamma='auto').fit(X_train)
    score_svm = clf.decision_function(X_test)#lower, more abnormal
    pr_oneclassSVM = precision_at_K(score_svm,label_test)
    #Isolation Forest
    clf = IsolationForest()
    clf.fit(X_train)
    score_if = clf.decision_function(X_test)#lower, more abnormal
    pr_iso_forest = precision_at_K(score_if,label_test)
    #Roundtrip
    py = RTM.estimate_py_with_IS(X_test,epoch,sd_y=best_sd,scale=best_scale,sample_size=5000,log=True,save=False)
    pr_Roundtrip = precision_at_K(py,label_test)
    print("The precision at K of Roundtrip model is %.4f"%pr_Roundtrip)    
    print("The precision at K of One-class SVM is %.4f"%pr_oneclassSVM)
    print("The precision at K of Isolation forest is %.4f"%pr_iso_forest)

def precision_at_K(score, label_test):
    rank = rankdata(score)
    nb_test = np.sum(label_test)
    precision = len([1 for item in zip(rank,label_test) if item[0]<=nb_test and item[1]==1])*1.0/nb_test
    return precision

def posterior_bayes():
    tst_data, tst_label, _ = RTM.y_sampler.load_all()
    tst_all = []
    tst_one_hot_all = []
    eval_idx = []
    for i in range(10):
        eval_idx += [j for j,item in enumerate(tst_label) if item==i][:100]
    #for idx in range(len(tst_data)):
    for idx in eval_idx:
        tst_all.append(np.tile(tst_data[idx],(10,1))) 
        tst_one_hot_all.append(np.eye(10))
    tst_all = np.concatenate(tst_all,axis=0)
    tst_one_hot_all = np.concatenate(tst_one_hot_all,axis=0)
    #For each test image, we evaluate the conditional density under 10 distinct labels
    py = RTM.estimate_py_with_IS(tst_all,tst_one_hot_all,epoch,sd_y=best_sd,scale=best_scale,sample_size=10000,log=True,save=False)
    py = py.reshape((-1,10))
    pre = np.argmax(py,axis=1)
    acc = accuracy_score(tst_label[eval_idx],pre)
    print('The test accuracy is %.4f.'%acc)


def visualize_img():
    if data == "mnist":
        for each in os.listdir(path):
            if each.startswith('py'):
                py = np.load('%s/%s'%(path, each))['arr_0']
                data_y_ = np.load('%s/%s'%(path, each))['arr_1']
                label_y = np.load('%s/%s'%(path, each))['arr_2']
                data_y_ = data_y_.reshape(data_y_.shape[0],28,28)
                for j in range(10):
                    idx = [i for i,item in enumerate(label_y) if item==j]
                    data_y_class_ = data_y_[idx]
                    combine = sorted(zip(data_y_class_[:25],py[25*j:25*(j+1)]),key=lambda x:x[1],reverse=True)
                    fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all',figsize=(10,10))
                    ax = ax.flatten()
                    for i in range(25):
                        ax[i].imshow(combine[i][0],plt.cm.gray)
                    ax[0].set_xticks([])
                    ax[0].set_yticks([])
                    plt.tight_layout(pad=1.08, h_pad=0.2, w_pad=0.2)  
                    plt.savefig('%s/mnist_pre_class_%d.png'%(path,j))
                    plt.show()
    else:
        for each in os.listdir(path):
            if each.startswith('py'):
                py = np.load('%s/%s'%(path, each))['arr_0']
                data_y_ = np.load('%s/%s'%(path, each))['arr_1']
                label_y = np.load('%s/%s'%(path, each))['arr_2']
                data_y_ = data_y_.reshape(data_y_.shape[0],32,32,3)
                for j in range(10):
                    idx = [i for i,item in enumerate(label_y) if item==j]
                    data_y_class_ = data_y_[idx]
                    combine = sorted(zip(data_y_class_[:25],py[25*j:25*(j+1)]),key=lambda x:x[1],reverse=True)
                    fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all',figsize=(10,10))
                    ax = ax.flatten()
                    for i in range(25):
                        ax[i].imshow(combine[i][0],plt.cm.gray)
                    ax[0].set_xticks([])
                    ax[0].set_yticks([])
                    plt.tight_layout(pad=1.08, h_pad=0.2, w_pad=0.2)  
                    plt.savefig('%s/cifar10_pre_class_%d.png'%(path,j))
                    plt.show()


def parse_params(path):
    exp_info = path.strip('/').split('/')[-1]
    timestamp = exp_info[12:27]
    x_dim = int(exp_info.split('=')[1].split('_')[0])
    y_dim = int(exp_info.split('=')[2].split('_')[0])
    print(x_dim, y_dim, timestamp)
    return x_dim, y_dim, timestamp


def find_y_sampler():
    global best_sd, best_scale
    if data == "indep_gmm":
        best_sd, best_scale = 0.05, 0.5
        ys = util.GMM_indep_sampler(N=20000, sd=0.1, dim=2, n_components=3, bound=1)

    elif data == "eight_octagon_gmm":
        best_sd, best_scale = 0.1, 0.5
        n_components = 8
        def cal_cov(theta,sx=1,sy=0.4**2):
            Scale = np.array([[sx, 0], [0, sy]])
            c, s = np.cos(theta), np.sin(theta)
            Rot = np.array([[c, -s], [s, c]])
            T = Rot.dot(Scale)
            Cov = T.dot(T.T)
            return Cov
        radius = 3
        mean = np.array([[radius*math.cos(2*np.pi*idx/float(n_components)),radius*math.sin(2*np.pi*idx/float(n_components))] for idx in range(n_components)])
        cov = np.array([cal_cov(2*np.pi*idx/float(n_components)) for idx in range(n_components)])
        ys = util.GMM_sampler(N=20000,mean=mean,cov=cov)
    
    elif data == "involute":
        best_sd, best_scale = 0.4, 0.5
        ys = util.Swiss_roll_sampler(N=20000)

    elif data == "uci_AReM":
        best_sd, best_scale = 0.1, 0.1
        ys = util.UCI_sampler('datasets/AReM/data.npy')
    elif data == "uci_CASP":
        best_sd, best_scale = 0.1, 0.1
        ys = util.UCI_sampler('datasets/Protein/data.npy')
    elif data == "uci_HEPMASS":
        best_sd, best_scale = 0.1, 0.1
        ys = util.hepmass_sampler()
    elif data == "uci_BANK":
        best_sd, best_scale = 0.1, 0.1
        ys = util.UCI_sampler('datasets/BANK/data.npy')
    elif data == "uci_YPMSD":
        best_sd, best_scale = 0.1, 0.1
        ys = util.UCI_sampler('datasets/YearPredictionMSD/data.npy')

    elif data == "odds_Shuttle":
        best_sd, best_scale = 0.05, 1
        ys = util.Outlier_sampler('datasets/ODDS/Shuttle/data.npz')
    elif data == "odds_Mammography":
        best_sd, best_scale = 0.05, 0.5
        ys = util.Outlier_sampler('datasets/ODDS/Mammography/data.npz')
    elif data == "odds_ForestCover":
        best_sd, best_scale = 0.1, 0.2
        ys = util.Outlier_sampler('datasets/ODDS/ForestCover/data.npz')

    elif data == "mnist":
        best_sd, best_scale = 0.1, 0.01
        ys = util.mnist_sampler()
    elif data == "cifar10":
        best_sd, best_scale = 0.1, 0.01
        ys = util.cifar10_sampler()
    else:
        print("Wrong data name!")
        sys.exit()
    return ys

def load_model(path, epoch):
    pool = util.DataPool()
    x_dim, y_dim, timestamp = parse_params(path)
    xs = util.Gaussian_sampler(mean=np.zeros(x_dim),sd=1.0)
    ys = find_y_sampler()

    if data == 'mnist' or data == 'cifar10':
        from main_density_est_img import RoundtripModel
        g_net = model.Generator_img(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=2,nb_units=256,dataset=data,is_training=False)
        h_net = model.Encoder_img(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=2,nb_units=256,dataset=data)
        dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
        dy_net = model.Discriminator_img(input_dim=y_dim,name='dy_net',nb_layers=2,nb_units=128,dataset=data)
        RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, data, pool, batch_size=64, nb_classes=10, alpha=10.0, beta=10.0, df=1, is_train=False)
    else:
        from main_density_est import RoundtripModel
        g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=512)
        h_net = model.Generator(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=10,nb_units=256)
        dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
        dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=4,nb_units=256)
        RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, data, pool, batch_size=64, alpha=10.0, beta=10.0, df=1, is_train=False)
    RTM.load(pre_trained=True, timestamp = timestamp, epoch = epoch)
    return RTM

    


if __name__=="__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='indep_gmm',help='name of data type')
    parser.add_argument('--epoch', type=int, default=100,help='which epoch to be loaded')
    parser.add_argument('--path', type=str, default='',help='path to ODDS predicted data')
    args = parser.parse_args()
    data = args.data
    epoch = args.epoch
    path = args.path
    model = importlib.import_module('model_img') if data=="mnist" or data=='cifar10' else importlib.import_module('model')

    RTM = load_model(path,epoch)
    if data == "indep_gmm":
        visualization_2d(-1.5, 1.5, -1.5, 1.5, 0.05, 0.5)
    elif data == "eight_octagon_gmm":
        visualization_2d(-5, 5, -5, 5, 0.1, 0.5)
    elif data == "involute":
        visualization_2d(-6, 5, -5, 5, 0.4, 0.5)
    elif data.startswith("odds"):
        odd_evaluate()
    elif data == "mnist":
        posterior_bayes()
    elif data == "cifar10":
        visualize_img()
    else:
        print("Wrong data name!")
