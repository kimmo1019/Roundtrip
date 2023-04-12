import yaml
from  Roundtrip import util
import argparse
from Roundtrip import Roundtrip,VariationalRoundtrip,RoundtripTV,RoundtripTV_img
from scipy.stats import pearsonr,spearmanr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    model = Roundtrip(params, random_seed=123)
    #model = VariationalRoundtrip(params, random_seed=123)
    #model = RoundtripTV(params, random_seed=123)
    #model  = RoundtripTV_img(params, random_seed=123)

    if params['dataset'] == "indep_gmm":
        xs = util.GMM_indep_sampler(N=20000, sd=0.1, dim=params['x_dim'], n_components=3, bound=1)
        import numpy as np
        np.save('test.npy',xs.X_train)
        sys.exit()
        model.train(data = xs.X_train, save_format='npy',n_iter=40000, batches_per_eval=10000)
        px_est = model.estimate_px_with_IS(xs.X_test,
                                        sd_x=params['sd_x'],
                                        scale=params['scale'],
                                        sample_size=params['sample_size'],log=False)
        px_true = xs.get_density(xs.X_test)
        print('Pearson correlation is %.3f, Spearman correlation is %.3f'%(pearsonr(px_est,px_true)[0],
                spearmanr(px_est,px_true)[0]))

    elif params['dataset'] == "eight_octagon_gmm":
        if not use_cv:
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
        xs = util.GMM_sampler(N=20000,mean=mean,cov=cov)
        model.train(data = xs.X_train, save_format='npy',n_iter=40000, batches_per_eval=20000)
        px_est = model.estimate_px_with_IS(xs.X_test,
                                        sd_x=params['sd_x'],
                                        scale=params['scale'],
                                        sample_size=params['sample_size'],log=False)
        px_true = xs.get_density(xs.X_test)
        print('Pearson correlation is %.3f, Spearman correlation is %.3f'%(pearsonr(px_est,px_true)[0],
                spearmanr(px_est,px_true)[0]))
    
    elif params['dataset'] == "involute":
        xs = util.Swiss_roll_sampler(N=20000)
        model.train(data = xs.X_train, save_format='npy',n_iter=40000, batches_per_eval=10000)
        px_est = model.estimate_px_with_IS(xs.X_test,
                                        sd_x=params['sd_x'],
                                        scale=params['scale'],
                                        sample_size=params['sample_size'],log=False)
        px_true = xs.get_density(xs.X_test)
        print('Pearson correlation is %.3f, Spearman correlation is %.3f'%(pearsonr(px_est,px_true)[0],
                spearmanr(px_est,px_true)[0]))
    
    elif params['dataset'] == "mnist":
        import tensorflow as tf
        import numpy as np
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        train_images = util.preprocess_images(train_images)
        model.train(data = train_images, save_format='npy',n_iter=100000, batches_per_eval=5000)

    elif params['dataset'] == "cifar10":
        import tensorflow as tf
        import numpy as np
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
        train_images = (train_images/255).astype('float32')
        test_images = (test_images/255).astype('float32')
        model.train(data = train_images, save_format='npy',n_iter=100000, batches_per_eval=5000)

    elif params['dataset'].startswith("uci"):
        if params['dataset'] == "uci_AReM":
            ys = util.UCI_sampler('datasets/AReM/data.npy')
        elif params['dataset'] == "uci_CASP":
            ys = util.UCI_sampler('datasets/Protein/data.npy')
        elif params['dataset'] == "uci_HEPMASS":
            ys = util.hepmass_sampler()
        elif params['dataset'] == "uci_BANK":
            ys = util.UCI_sampler('datasets/BANK/data.npy')
        elif params['dataset'] == "uci_YPMSD":
            ys = util.UCI_sampler('datasets/YearPredictionMSD/data.npy')
        else:
            print("Wrong UCI data name!")
            sys.exit()

    elif params['dataset'].startswith("odds"):
        if params['dataset'] == "odds_Shuttle":
            if not use_cv:
                best_sd, best_scale = 0.1, 0.1
            ys = util.Outlier_sampler('datasets/ODDS/Shuttle/data.npz')
        elif params['dataset'] == "odds_Mammography":
            if not use_cv:
                best_sd, best_scale = 0.05, 0.01
            ys = util.Outlier_sampler('datasets/ODDS/Mammography/data.npz')
        elif params['dataset'] == "odds_ForestCover":
            if not use_cv:
                best_sd, best_scale = 0.1, 0.2
            ys = util.Outlier_sampler('datasets/ODDS/ForestCover/data.npz')
        else:
            print("Wrong ODDS data name!")
            sys.exit()
    
    else:
        print("Wrong data name!")
        sys.exit()

            
