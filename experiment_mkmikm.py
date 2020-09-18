import time
import os
import numpy as np
from mkmikm import multiple_kernel_multi_instance_kmeans, gaussian
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import pairwise_distances


list_sigmas = np.array([1e-2, 5e-2, 1e-1, 1, 10, 50, 100])
m = list_sigmas.shape[0]
n_jobs = 1

dataset_dir = 'datasets-npz/'
datasets = ['digits', 'olivetti', 'umist', 'usps', 'coil20_s32x32', 'coil100_s32x32',
    'yaleb_s32x32', 'mnist', 'fashion_mnist', 'stl10', 'cifar10_gray', 'cifar100_gray']

#if not os.path.exists('res_ws_mkkm5_1'):
#    os.makedirs('res_ws_mkkm5_1')
#fw = open('res_ws_mkkm5_1/res1.txt', 'w')
#fw.close()

for i in range(len(datasets)):
    tmp = np.load(dataset_dir + datasets[i] + '.npz', allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    tmp = None
    n_clusters = len(np.unique(y))
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    tmp = X.max(axis=0)
    tmp[tmp==0] = 1
    X = X / tmp
    print(datasets[i], X.shape, n_clusters)

    #if not os.path.exists('res_ws_mkkm5_1/'+datasets[i]):
    #    os.makedirs('res_ws_mkkm5_1/'+datasets[i])

    n_set = 10
    bag_ari = np.zeros((n_set))
    data_ari = np.zeros((n_set))

    for i_set in range(n_set):
        print(i_set+1,' / 10')
        tmp = np.load('data_bags_npz/'+datasets[i]+'/'+ datasets[i] +'_set'+ str(i_set) +'.npz')
        bagX = X[tmp['bagX_idxs']]
        bagy = tmp['bagy']
        bag_sidx = tmp['bag_sidx']
        bag_len = np.hstack((bag_sidx[1:] - bag_sidx[0:-1], bagX.shape[0]-bag_sidx[-1]))
        all_instances_y = tmp['all_instances_y']
        tmp = None

        centers, H, w, costs = multiple_kernel_multi_instance_kmeans(
            bagX, bagy, bag_sidx, bag_len, list_sigmas=list_sigmas, max_iter=100, n_init=5, n_jobs=n_jobs
        )

        bagmem = H.argmax(axis=0)

        #np.save('res_ws_mkkm5_1/'+datasets[i]+'/'+datasets[i] +'_bagmem_set'+ str(i_set) +'.npy', bagmem)
        #np.save('res_ws_mkkm5_1/'+datasets[i]+'/'+datasets[i] +'_w_set'+ str(i_set) +'.npy', w)
        #np.save('res_ws_mkkm5_1/'+datasets[i]+'/'+datasets[i] +'_centers_set'+ str(i_set) +'.npy', centers)
        #np.save('res_ws_mkkm5_1/'+datasets[i]+'/'+datasets[i] +'_costs_set'+ str(i_set) +'.npy', costs)

        bag_ari[i_set] = ARI(all_instances_y, bagmem)

        sqdist = pairwise_distances(bagX, Y=bagX, n_jobs=n_jobs)
        Kcomb = np.zeros(sqdist.shape)
        for j in range(m):
            Kcomb += gaussian(sqdist, sigma=list_sigmas[j]) * (w[j] ** 2)
        term3 = np.zeros((n_clusters))
        for j in range(n_clusters):
            term3[j] = ((Kcomb * H[j]).sum(axis=1) * H[j]).sum() / ((H[j].sum() * (w ** 1).sum()) ** 2)
        sqdist = pairwise_distances(X, Y=bagX, n_jobs=n_jobs)
        Kcomb = np.zeros(sqdist.shape)
        for j in range(m):
            Kcomb += gaussian(sqdist, sigma=list_sigmas[j]) * (w[j] ** 2)
        dist = np.zeros((n_clusters, X.shape[0]))
        for j in range(n_clusters):
            dist[j] = (- 2 * (
                (H[j] * Kcomb).sum(axis=1) / (H[j].sum() * (w ** 1).sum())
            ) + term3[j])
        datamem = dist.argmin(axis=0)

        #np.save('res_ws_mkkm5_1/'+datasets[i]+'/'+datasets[i] +'_datamem_set'+ str(i_set) +'.npy', datamem)

        data_ari[i_set] = ARI(y, datamem)

        print('BagARI, DataARI:', bag_ari[i_set], data_ari[i_set])

        #with open('res_ws_mkkm5_1/res1.txt', 'a') as fa:
        #    fa.write(
        #        datasets[i] + ' ' + str(i_set) + ' ' + str(bag_ari[i_set]) + ' '
        #        + str(data_ari[i_set]) + '\n'
        #    )
    #with open('res_ws_mkkm5_1/res1.txt', 'a') as fa:
    #    fa.write(
    #        datasets[i] + ' Avg. ' + str(bag_ari.mean()) + ' ' + str(data_ari.mean())
    #        + '\n'
    #    )
    break
