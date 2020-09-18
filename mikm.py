import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


def multi_instance_kmeans(X, Y, bag_sidx, bag_len, max_iter=100, n_init=1):
    n = X.shape[0]
    k = Y.shape[1]

    min_cost = +np.inf
    for _ in range(n_init):
        H = np.zeros((k, n), dtype=np.bool)
        H[np.random.randint(0, k, X.shape[0]), np.arange(n)] = 1
        for _ in range(max_iter):
            Z = np.zeros((k, X.shape[1]), dtype=np.float32)
            for j in range(k):
                if H[j].sum() > 0:
                    Z[j] = X[H[j]].mean(axis=0)

            dist = pairwise_distances(X, Y=Z, n_jobs=4)

            prevH = np.array(H, dtype=np.bool)
            H = np.zeros(H.shape, dtype=np.bool)
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
                    idx2 = dist[optimal_assign_idx].argsort()[::-1][0:pi]
                    tmp[optimal_assign_idx[0][idx2], optimal_assign_idx[1][idx2]] = 1
                    H[nzY, bag_sidx[j]:bag_sidx[j]+bag_len[j]] = tmp.T

            if not (prevH ^ H).sum():
                break

        cost = (H * dist.T).sum()
        if cost < min_cost:
            min_cost = cost
            mincost_H = np.array(H)

    return mincost_H


if __name__ == '__main__':
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

        H = multi_instance_kmeans(bagX, bagy, bag_sidx, bag_len, max_iter=100, n_init=5)
        centers = np.linalg.pinv(H.T).dot(bagX)

        pred_y = H.argmax(axis=0)
        from sklearn.metrics import adjusted_rand_score as ARI
        bag_ari = ARI(true_y, pred_y)
        print('weak supervision ARI:', bag_ari)
        mean_bag_ari += bag_ari

        pred_y2 = cdist(centers, X, metric='sqeuclidean').argmin(axis=0)
        data_ari = ARI(y, pred_y2)
        print('transfer clustering ARI:', data_ari)
        mean_data_ari += data_ari

    print('Avg. weak supervision ARI:', mean_bag_ari/10)
    print('Avg. transfer clustering ARI:', mean_data_ari/10)
