import numpy as np
import time

class PQ(object):
    def __init__(self, subdims, num_subvecs, num_centers):
        self.centers = np.zeros((num_subvecs, num_centers, subdims))
        self.db_nn_idxs = None
        self.subdims = subdims
        self.num_subvecs = num_subvecs
        self.num_centers = num_centers
        self.sdc_table =  None

    def train(self, X):
        '''
        Input:
            X: N*C array
        '''
        assert X.shape[1] == self.num_subvecs * self.subdims
        from sklearn.cluster import KMeans
        cluster = KMeans(self.num_centers, n_jobs = 8)
        for i in range(self.num_subvecs):
            subvecs = X[:, i*self.subdims : (i+1)*self.subdims]
            cluster.fit(subvecs)
            self.centers[i, :, :] += cluster.cluster_centers_
            print "Training {}/{}".format(i, self.num_subvecs)
        nn_idxs, nn_dists = self._predict(X)
        self.db_nn_idxs = nn_idxs
        self.sdc_table = self._calc_sdc_table()

    def _predict(self, X):
        '''
        Input:
            X: N*C array
        Output:
            nn_idxs: N*self.num_subvecs int64 array. The nn indexs for each subvector,
        '''
        assert X.shape[1] == self.num_subvecs * self.subdims
        nn_idxs = []
        nn_dists = []
        for i in range(self.num_subvecs):
            subvecs = X[:, i*self.subdims : (i+1) * self.subdims]
            centers = self.centers[i, :, :]
            dists = []
            for center in centers:
                #print center.shape, subvecs.shape
                dist = np.sum((center - subvecs)**2, axis=1, keepdims=1)    #N*1 array
                dists.append(dist)
            dists = np.hstack(dists)    #N*num_centers array
            idxs = np.argmin(dists, axis=1).reshape(-1, 1) #N*1 array
            nn_idxs.append(idxs)
            nn_dists.append(dists[np.arange(0, dists.shape[0]), idxs.reshape(-1)].reshape(-1, 1))
        nn_idxs = np.hstack(nn_idxs)    #N*num_subvecs array
        nn_dists = np.hstack(nn_dists)
        return nn_idxs, nn_dists

    def ADC(self, query, n_keep=-1):
        '''
        Input:
            query: 1*C array
            n_keep: int dtype. the number of returned retrieval results.
        Output:
            retrieval_idx: ( num_db) array, the index of retrieval result in db
            retrieval_dist: cooresponding squared l2 distances
        '''
        assert query.shape[1] == self.subdims * self.num_subvecs
        assert query.shape[0] == 1
        table = np.zeros((self.num_subvecs, self.num_centers))    #num_subvecs * num_centers
        q_rsp = query.reshape(self.num_subvecs, self.subdims)
        s1 = time.time()
        for i in range(self.num_subvecs):
            subvec = q_rsp[i:i+1, :]
            dists = self._l2_dist(subvec - self.centers[i, :, :]).reshape(-1)
            table[i, :] += dists
        e1 = time.time()
        
        dists = np.zeros(self.db_nn_idxs.shape[0])
        s2= time.time()
        for i in range(self.num_subvecs):
            tmp = table[i, self.db_nn_idxs[:, i]]
            dists += tmp
        '''
        for i in range(self.db_nn_idxs.shape[0]):
            nn_idxs = self.db_nn_idxs[i].reshape(-1)
            #dist = np.sum(table[np.arange(table.shape[0]), nn_idxs])
            for j in range(self.num_subvecs):
                dists[i] += table[j, self.db_nn_idxs[i, j]]
        '''
        e2= time.time()
        s3= time.time()
        retrieval_idx = np.argsort(dists)[:n_keep]
        retrieval_dist = dists[retrieval_idx]
        e3= time.time()
        #print "build search table ", e1 - s1
        #print "search ", e2 - s2
        #print "rank ", e3 - s3
        return retrieval_idx, retrieval_dist

    def SDC(self, query, n_keep=-1):
        assert query.shape[1] == self.subdims * self.num_subvecs
        assert query.shape[0] == 1
        if self.sdc_table is  None:
            self._calc_sdc_table()  #sdc table : num_subvecs* num_centers * num_centers
        q_rsp = query.reshape(self.num_subvecs, self.subdims)
        dists = np.zeros(self.db_nn_idxs.shape[0])
        
        for i in range(self.num_subvecs):
            subvec = q_rsp[i:i+1, :]
            subdists = self._l2_dist(subvec - self.centers[i, :, :], keepdims=False)
            nn_idx = np.argmin(subdists)
            tmp = self.sdc_table[i, nn_idx, self.db_nn_idxs[:, i]]
            dists += tmp

        '''
        q_nn_idxs = []
        for i in range(self.num_subvecs):
            subvec = q_rsp[i:i+1, :]
            dists = self._l2_dist(subvec - self.centers[i, :, :], keepdims=False)
            q_nn_idxs.append(np.argmin(dists))
        q_nn_idxs = np.hstack(q_nn_idxs)    #1*num_subvecs shape

        dists = []
        for i in range(self.db_nn_idxs.shape[0]):
            dist = np.sum(self.sdc_table[np.arange(self.num_subvecs), q_nn_idxs, self.db_nn_idxs[i,:]])
            dists.append(dist)
        dists = np.hstack(dists)
        '''
        retrieval_idx = np.argsort(dists)[:n_keep]
        retrieval_dist = dists[retrieval_idx]
        return retrieval_idx, retrieval_dist

    def _l2_dist(self, x, keepdims=True):
        return np.sum(x**2, axis=-1, keepdims=keepdims)

    def _calc_sdc_table(self):
        dists = np.zeros((self.num_subvecs, self.num_centers, self.num_centers))
        for i in range(self.num_subvecs):
            centers1 = self.centers[i, :, :].reshape(self.num_centers, 1, self.subdims)
            centers2 = self.centers[i, :, :].reshape(1, self.num_centers, self.subdims)
            dists[i, :, :] += self._l2_dist(centers1 - centers2, keepdims=False)
        return dists
        

def brute_force_l2(query, db):
    dists = np.sum((query - db)**2, axis=1)
    idxs = np.argsort(dists)
    return idxs, dists

def brute_force_cosine(query, db):
    cosine = query.dot(db.T)
    #cosine = query_norm.dot(db_norm.T)
    idxs = np.argsort(-cosine)
    return idxs, -cosine

if __name__ == "__main__":
    import os
    dims = 1024
    num_subvecs = 8
    subdims = dims/ num_subvecs
    num_centers = 256
    center_file = "./center.npy"
    db_nn_table_file = "db_table.npy"
    sdc_table_file = "sdc_table.npy"

    X = np.random.randn(10000, dims)

    pq = PQ(subdims, num_subvecs, num_centers)
    if os.path.exists(db_nn_table_file) and os.path.exists(center_file) and os.path.exists(sdc_table_file):
        pq.centers = np.load(center_file)
        pq.db_nn_idxs = np.load(db_nn_table_file)
        pq.sdc_table = np.load(sdc_table_file)
    else:
        pq.train(X)
        np.save(center_file, pq.centers)
        np.save(db_nn_table_file, pq.db_nn_idxs)
        np.save(sdc_table_file, pq.sdc_table)
    query = X[0:1, :]
    s1 = time.time()
    bf_idx, bf_dist = brute_force_l2(query, X)
    e1 = time.time()

    s2 = time.time()
    sdc_pq_idx, sdc_pq_dist = pq.SDC(query)
    e2 = time.time()

    adc_pq_idx, adc_pq_dist = pq.ADC(query)
    s3 = time.time()
    adc_pq_idx, adc_pq_dist = pq.ADC(query)
    e3 = time.time()

    query_norm  = query / np.sqrt(np.sum( query **2, axis=1, keepdims=1) + 1e-10)
    X_norm = X / np.sqrt(np.sum(X**2, axis=1, keepdims=1) + 1e-10)
    s4 = time.time()
    bf_idx, bf_dist = brute_force_cosine(query_norm, X_norm)
    e4 = time.time()

    print "SDC ", e2 - s2
    print "ADC ", e3 - s3
    print "Brute force l2", e1 - s1
    print "Brute force cosine", e4 - s4

