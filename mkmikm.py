import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed


def gaussian(sqdist, sigma=1):
    K = np.exp(-sqdist / (2 * (sigma * sqdist.max()) ** 2))
    tmp = K.max()
    if tmp == 0:
        tmp = 1
    return K / tmp


def get_bag_mem(dist, Y, bag_sidx, bag_len):
    H = np.zeros(dist.T.shape, dtype=np.bool)
    for j in range(len(bag_len)):
        nzY = Y[j,:]!=0
        if nzY.sum() == 1:
            H[
                nzY,
                bag_sidx[j]+dist[
                    bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY
                ].argmin() # attempt to get argmin of an empty sequence
            ] = 1
        elif bag_len[j] == 1:
            H[
                nzY[dist[bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY].argmin()],
                bag_sidx[j]:bag_sidx[j]+bag_len[j]
            ] = 1
        else:
            tmp = np.zeros((bag_len[j], nzY.sum()), dtype=np.bool)
            optimal_assign_idx = linear_sum_assignment(
                dist[bag_sidx[j]:bag_sidx[j]+bag_len[j], nzY]
            )
            pi = min(bag_len[j], nzY.sum())
            idx = dist[optimal_assign_idx].argsort()[::-1][0:pi]
            tmp[optimal_assign_idx[0][idx], optimal_assign_idx[1][idx]] = 1
            H[nzY, bag_sidx[j]:bag_sidx[j]+bag_len[j]] = tmp.T
    return H


def multiple_kernel_multi_instance_kmeans(X, Y, bag_sidx, bag_len, list_sigmas, max_iter=100, n_init=1, n_jobs=1):
    n = X.shape[0]
    k = Y.shape[1]
    m = len(list_sigmas)

    min_cost = +np.inf
    for _ in range(n_init):
        centers = X[np.random.permutation(X.shape[0])[0:k]]
        H = np.zeros((k, n), dtype=np.bool)
        H[np.random.randint(0, k, n), np.arange(n)] = 1
        w = np.ones((m), dtype=np.float32) / m
        sqdist = pairwise_distances(centers, Y=X, n_jobs=n_jobs)
        K = np.array(Parallel(n_jobs=n_jobs)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
        dist = ((w ** 2)[:,None,None] * (2 - K)).sum(axis=0)
        cost = np.zeros((max_iter)) - 1

        for v_iter in range(max_iter):
            # Update H
            prevH = np.array(H)
            H = get_bag_mem(dist.T, Y, bag_sidx, bag_len)

            # Update centers
            dist2 = (((w ** 2) / ((list_sigmas * sqdist.max()) ** 2))[:,None,None] * K).sum(axis=0)
            for j in range(k):
                tmp = H[j] * dist2[j]
                centers[j] = (tmp[:,None] * X).sum(axis=0) / tmp.sum()

            # Update K
            sqdist = pairwise_distances(centers, Y=X, n_jobs=n_jobs)
            K = np.array(Parallel(n_jobs=n_jobs)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))

            # Update w
            a = 1 / np.fmax((H[None, :, :] * (2 - K)).sum(axis=-1).sum(axis=-1), 1e-6)
            w = a / a.sum()

            dist = ((w ** 2)[:,None,None] * (2 - K)).sum(axis=0)
            cost[v_iter] = (H * dist).sum()

            if not (prevH ^ H).sum():
                print('break at', v_iter)
                break

        if cost[min(v_iter, max_iter)] < min_cost:
            min_cost = cost[min(v_iter, max_iter)]
            mincost_centers = np.array(centers)
            mincost_H = np.array(H)
            mincost_w = np.array(w)
            saved_costs = np.array(cost)

    return mincost_centers, mincost_H, mincost_w, saved_costs


if __name__ == '__main__':
    n_jobs = 1

    list_sigmas = np.array([1e-2, 5e-2, 1e-1, 1, 10, 50, 100])
    m = list_sigmas.shape[0]

    from sklearn.datasets import load_digits
    from sklearn.metrics import adjusted_rand_score as ARI

    dataset = 'digits'
    X, y = load_digits(return_X_y=True)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if (X < 0).sum() > 1:
        X = X - X.min()
    tmp = X.max(axis=0)
    tmp[tmp==0] = 1
    X = X / tmp

    mean_bag_ari = 0
    mean_data_ari = 0

    for i2 in range(10):
        print(i2+1, '/ 10')
        tmp = np.load('data_bags_npz/'+dataset+'/'+dataset+'_set'+str(i2)+'.npz')
        bagX = X[tmp['bagX_idxs']]
        n = bagX.shape[0]
        bagy = tmp['bagy']
        n_clusters = bagy.shape[1]
        bag_sidx = tmp['bag_sidx']
        bag_len = np.hstack((bag_sidx[1:] - bag_sidx[0:-1], n - bag_sidx[-1]))
        true_y = tmp['all_instances_y']

        centers, H, w, costs = multiple_kernel_multi_instance_kmeans(bagX, bagy, bag_sidx, bag_len, list_sigmas=list_sigmas, max_iter=100, n_init=5, n_jobs=n_jobs)

        pred_y = H.argmax(axis=0)

        bag_ari = ARI(true_y, pred_y)
        print('weak supervision ARI:', bag_ari)
        mean_bag_ari += bag_ari

        sqdist = pairwise_distances(bagX, Y=bagX, n_jobs=n_jobs)
        Kcomb = (
            np.array(Parallel(n_jobs=n_jobs)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
            * (w[:,None,None] ** 2)
        ).sum(axis=0)
        term3 = np.zeros((n_clusters))
        for j in range(n_clusters):
            term3[j] = ((Kcomb * H[j]).sum(axis=1) * H[j]).sum() / ((H[j].sum() * (w ** 1).sum()) ** 2)
        sqdist = pairwise_distances(X, Y=bagX, n_jobs=n_jobs)
        K = np.array(Parallel(n_jobs=n_jobs)(delayed(gaussian)(sqdist, sigma=list_sigmas[i]) for i in range(m)))
        dist = np.zeros((n_clusters, X.shape[0]))
        for j in range(n_clusters):
            dist[j] = (1 - 2 * (
                (H[j] * (K * w[:,None,None] ** 2).sum(axis=0)).sum(axis=1) / (H[j].sum() * (w ** 1).sum())
            ) + term3[j])
        data_ari = ARI(y, dist.argmin(axis=0))
        print('transfer clustering ARI:', data_ari)
        mean_data_ari += data_ari

    print('Avg. weak supervision ARI:', mean_bag_ari/10)
    print('Avg. transfer clustering ARI:', mean_data_ari/10)
