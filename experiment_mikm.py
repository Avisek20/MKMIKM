import time
import os
import numpy as np
from mikm import multi_instance_kmeans
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


n_jobs = 1

dataset_dir = 'datasets-npz/'
datasets = ['digits', 'olivetti', 'umist', 'coil20_s32x32', 'coil100_s32x32',
    'yaleb_s32x32', 'usps', 'mnist', 'fashion_mnist', 'stl10', 'cifar10_gray',
    'cifar100_gray']

#if not os.path.exists('res_exp_wskm'):
#    os.makedirs('res_exp_wskm')
#fw = open('res_exp_wskm/res1.txt', 'w')
#fw.close()

for i in range(len(datasets)):
    tmp = np.load(dataset_dir + datasets[i] + '.npz', allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    del tmp
    k = len(np.unique(y))
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    print(datasets[i], X.shape, k)

    bagX_idxs = []
    bagy = []
    bag_sidx = []
    bag_len = []
    all_instances_y = []
    n_set = 10
    for i_set in range(n_set):
        tmp = np.load('data_bags_npz/'+datasets[i]+'/'+ datasets[i] +'_set'+ str(i_set) +'.npz')
        bagX_idxs.append(tmp['bagX_idxs'])
        bagy.append(tmp['bagy'])
        bag_sidx.append(tmp['bag_sidx'])
        bag_len.append(np.hstack((bag_sidx[-1][1:] - bag_sidx[-1][0:-1], bagX_idxs[-1].shape[0]-bag_sidx[-1][-1])))
        all_instances_y.append(tmp['all_instances_y'])
        #print(X[bagX_idxs[-1]].shape, bagy[-1].shape, bag_sidx[-1].shape, bag_len[-1].shape, all_instances_y[-1].shape, len(np.unique(all_instances_y[-1])))

    results = Parallel(n_jobs=n_jobs)(delayed(multi_instance_kmeans)(
        X=X[bagX_idxs[i_set]], Y=bagy[i_set], bag_sidx=bag_sidx[i_set], bag_len=bag_len[i_set],
        max_iter=100, n_init=5
    ) for i_set in range(n_set))

    #if not os.path.exists('res_exp_wskm/'+datasets[i]):
    #    os.makedirs('res_exp_wskm/'+datasets[i])

    bag_ari = np.zeros((n_set))
    data_ari = np.zeros((n_set))

    for i_set in range(n_set):
        H = results[i_set]
        centers = np.zeros((k, X.shape[1]))
        for j in range(k):
            centers[j] = X[bagX_idxs[i_set]][H[j]].mean(axis=0)
        bagmem = H.argmax(axis=0)

        #np.save('res_exp_wskm/'+datasets[i]+'/'+datasets[i] +'_bagmem_set'+ str(i_set) +'.npy', bagmem)

        datamem = pairwise_distances(centers, Y=X, n_jobs=n_jobs).argmin(axis=0)

        #np.save('res_exp_wskm/'+datasets[i]+'/'+datasets[i] +'_datamem_set'+ str(i_set) +'.npy', datamem)

        bag_ari[i_set] = ARI(all_instances_y[i_set], bagmem)
        data_ari[i_set] = ARI(y, datamem)

        print('BagARI, DataARI:', bag_ari[i_set], data_ari[i_set])

        #with open('res_exp_wskm/res1.txt', 'a') as fa:
        #    fa.write(
        #        datasets[i] + ' ' + str(i_set) + ' ' + str(bag_ari[i_set]) + ' '
        #        + str(data_ari[i_set]) '\n'
        #    )
    break
