import numpy as np

#The Optimized Product Quantization: the Non-Parametric solution
class NPOPQ(object):
    def __init__(self, subdim, num_subvec, num_center):
        self.centers = np.random.randn(num_center, num_subvec*subdim)
        self.R = self._gen_orthogonal_matrix(subdim*num_subvec)
        self.db_nn_idxs = None
        self.sdc_table = np.zeros((num_center, num_center))

        self.num_subvec = num_subvec
        self.subdim = subdim
        self.num_center = num_center

    def _gen_orthogonal_matrix(self, dim):
        mat = np.random.randn(dim, dim)
        mat, _, _ = np.linalg.svd(mat)
        return mat

    def train(self, X, num_iters, max_iter=300):
        '''
        Input:
            X: N *(num_subvec*subdim) array
            num_iters: the number of iteration
        '''
        from sklearn.cluster import KMeans
        self.centers = X[np.random.choice(X.shape[0], self.num_center), :].dot(self.R)
        self.db_nn_idxs = np.zeros((X.shape[0], self.num_subvec), dtype=np.int)
        for i in range(num_iters):
            X_hat = X.dot(self.R)
            #Stage 1, fix R and update centers
            Y = np.zeros_like(X_hat)
            for j in range(self.num_subvec):
                subvecs = X_hat[:, j*self.subdim : (j+1)*self.subdim]
                subcenters = self.centers[:, j*self.subdim : (j+1)*self.subdim].reshape(self.num_center, self.subdim)

                cluster = KMeans(n_clusters=self.num_center, init=subcenters, n_init=1, n_jobs=8, max_iter=max_iter)
                idxs = cluster.fit_predict(subvecs)

                self.centers[:, j*self.subdim : (j+1)*self.subdim] = cluster.cluster_centers_
                Y[:, j*self.subdim : (j+1)*self.subdim] += cluster.cluster_centers_[idxs, :]    #assign the nn center for each sample
                self.db_nn_idxs[:, j] = idxs
                print "Iter: {}, {}th center training done".format(i, j)

            #Stage 2, fix centers and update R, min || XR - Y||. the solution is the U.dot(VT) where U, S, V = svd(XT.dot(Y) 
            U, S, V = np.linalg.svd((X.T).dot(Y))
            self.R = U.dot(V)
            loss1 = np.mean((X_hat - Y)**2)
            loss2 = np.mean((X.dot(self.R) - Y)**2)
            print "loss1: ", loss1, " loss2: ", loss2 #, np.sum(self.R.dot(self.R.T))
        self.sdc_table = self._calc_sdc_table()

    def _calc_sdc_table(self):
        dists = np.zeros((self.num_subvec, self.num_center, self.num_center))
        for i in range(self.num_subvec):
            centers1 = self.centers[:, i*self.subdim : (i+1)*self.subdim].reshape(self.num_center, 1 ,self.subdim)
            centers2 = self.centers[:, i*self.subdim : (i+1)*self.subdim].reshape(1, self.num_center, self.subdim)
            dists[i, :, :] += self._l2_dist(centers1 - centers2, keepdims=False)
        return dists

    def _l2_dist(self, x, keepdims=True):
        return np.sum(x**2, axis=-1, keepdims=keepdims)

    def ADC(self, query, n_keep=-1):
        '''
        Input:
            query: 1*C array
            n_keep: int dtype. the number of returned retrieval results.
        Output:
            retrieval_idx: ( num_db) array, the index of retrieval result in db
            retrieval_dist: cooresponding squared l2 distances
        '''
        assert query.shape[1] == self.subdim * self.num_subvec
        assert query.shape[0] == 1
        query_hat = query.dot(self.R)
        table = np.zeros((self.num_subvec, self.num_center))    #num_subvecs * num_centers
        q_rsp = query_hat.reshape(self.num_subvec, self.subdim)
        for i in range(self.num_subvec):
            subvec = q_rsp[i:i+1, :]
            subcenters = self.centers[:, i*self.subdim : (i+1) * self.subdim]
            dists = self._l2_dist(subvec - subcenters).reshape(-1)
            table[i, :] += dists
        dists = np.zeros(self.db_nn_idxs.shape[0])
        for i in range(self.num_subvec):
            tmp = table[i, self.db_nn_idxs[:, i]]
            dists += tmp
        retrieval_idx = np.argsort(dists)[:n_keep]
        retrieval_dist = dists[retrieval_idx]
        return retrieval_idx, retrieval_dist

    def SDC(self, query, n_keep=-1):
        assert query.shape[1] == self.subdim * self.num_subvec
        assert query.shape[0] == 1
        query_hat = query.dot(self.R)
        q_rsp = query_hat.reshape(self.num_subvec, self.subdim)
        dists = np.zeros(self.db_nn_idxs.shape[0])
        
        for i in range(self.num_subvec):
            subvec = q_rsp[i:i+1, :]
            subcenters = self.centers[:, i*self.subdim : (i+1)*self.subdim]
            subdists = self._l2_dist(subvec - subcenters, keepdims=False)
            nn_idx = np.argmin(subdists)
            tmp = self.sdc_table[i, nn_idx, self.db_nn_idxs[:, i]]
            dists += tmp
        retrieval_idx = np.argsort(dists)[:n_keep]
        retrieval_dist = dists[retrieval_idx]
        return retrieval_idx, retrieval_dist

def brute_force_l2(query, db):
    dists = np.sum((query - db)**2, axis=1)
    idxs = np.argsort(dists)
    return idxs, dists

if __name__ == "__main__":
    num_center = 4
    num_subvec = 4
    subdim = 4
    '''
    num_center = 256
    num_subvec = 8
    subdim =  128
    '''

    X = np.random.randn(100, num_subvec * subdim)
    print "Start 1"
    opq3 = NPOPQ(subdim, num_subvec, num_center)
    opq3.train(X, 10, 1)

    query = X[0:1, :]
    adc_pq_idx, adc_pq_dist = opq3.ADC(query)
    sdc_pq_idx, sdc_pq_dist = opq3.SDC(query)
    bf_idx, bf_dist = brute_force_l2(query, X)
    
    '''
    print "Start 2"
    opq2 = NPOPQ(subdim, num_subvec, num_center)
    opq2.train(X, 10, 10)

    print "Start 3"
    opq = NPOPQ(subdim, num_subvec, num_center)
    opq.train(X, 20, 1)
    '''
