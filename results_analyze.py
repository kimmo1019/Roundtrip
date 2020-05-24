import numpy as np
import sys, os
import argparse
import importlib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from main_density_est import RoundtripModel
import util
from scipy.stats import rankdata
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n=100):
    grid_x1 = np.linspace(x1_min, x1_max, n)
    grid_x2 = np.linspace(x2_min, x2_max, n)
    v1,v2 = np.meshgrid(grid_x1,grid_x2)
    data_grid = np.vstack((v1.ravel(),v2.ravel())).T
    return v1, v2, data_grid


def visualization_2d(x1_min, x1_max, x2_min, x2_max, sd_y, scale, n=100):
    v1, v2, data_grid = create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n)
    py = RTM.estimate_py_with_IS(data_grid,0,sd_y=sd_y,scale=scale,sample_size=40000,log=False,save=False)
    py = py.reshape((n,n))
    plt.figure()
    plt.rcParams.update({'font.size': 22})
    plt.imshow(py, extent=[v1.min(), v1.max(), v2.min(), v2.max()],
cmap='Blues', alpha=0.9)
    plt.colorbar()
    plt.savefig('%s/2d_grid_density_pre.png'%save_dir)
    plt.close()

def odd_evaluate():
    X_test = ys.X_test
    X_train = ys.X_train
    label_test = ys.label_test
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
    for each in os.listdir(path):
        if each.startswith('py_est_at_epoch%d'%epoch):
            py = np.load('%s/%s'%(path,each))['arr_0']
            pr_Roundtrip = precision_at_K(py,label_test)        
    print("The precision at K of Roundtrip model is %.4f"%pr_Roundtrip)
    print("The precision at K of One-class SVM is %.4f"%pr_oneclassSVM)
    print("The precision at K of Isolation forest is %.4f"%pr_iso_forest)

def precision_at_K(score, label_test):
    rank = rankdata(score)
    nb_test = np.sum(label_test)
    precision = len([1 for item in zip(rank,label_test) if item[0]<=nb_test and item[1]==1])*1.0/nb_test
    return precision
    
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




if __name__=="__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='indep_gmm')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100,help='which epoch to be loaded')
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--path', type=str, default='',help='path to ODDS predicted data')
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--df', type=float, default=1,help='degree of freedom of student t distribution')
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    alpha = args.alpha
    beta = args.beta
    epoch = args.epoch
    df = args.df
    is_train = args.train
    path = args.path
    timestamp = args.timestamp

    save_dir = 'data/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(timestamp, data, x_dim, y_dim, alpha, beta)
    g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=512)
    h_net = model.Generator(input_dim=y_dim,output_dim = x_dim,name='h_net',nb_layers=10,nb_units=256)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=128)
    dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=4,nb_units=256)
    pool = util.DataPool()
    xs = util.Gaussian_sampler(N=5000,mean=np.zeros(x_dim),sd=1.0)
    ys = util.GMM_indep_sampler(N=20000, sd=0.1, dim=y_dim, n_components=3, bound=1)
    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, data, pool, batch_size, alpha, beta, df, is_train)
    RTM.load(pre_trained=False, timestamp = timestamp, epoch = epoch)
    if data == "indep_gmm":
        visualization_2d(-1.5, 1.5, -1.5, 1.5, 0.05, 0.5)
    elif data == "eight_octagon_gmm":
        visualization_2d(-5, 5, -5, 5, 0.1, 0.5)
    elif data == "involute":
        visualization_2d(-6, 5, -5, 5, 0.4, 0.5)
    elif data.startswith("odds"):
        if data == "odds_Shuttle":
            ys = util.Outlier_sampler('datasets/ODDS/Shuttle/data.npz')
            odd_evaluate()
        elif data == "odds_Mammography":
            ys = util.Outlier_sampler('datasets/ODDS/Mammography/data.npz')
            odd_evaluate()
        elif data == "odds_ForestCover":
            ys = util.Outlier_sampler('datasets/ODDS/ForestCover/data.npz')
            odd_evaluate()
        else:
            print("Wrong ODDS data name!")
            sys.exit()
    elif data == "mnist":
        visualize_img()
    elif data == "cifar10":
        visualize_img()
    else:
        print("Wrong data name!")
