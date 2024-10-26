import time
import numpy as np
from f1.clustering.utility import distance_cdist
from f1.clustering.cmeans.fcm import Dfcm


class Dpfcm(Dfcm):
    def __init__(self, n_clusters: int, a: float, b: float, eta: float, K: float = 1, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000, index: int = 0, metric: str = 'euclidean'):  # euclidean|chebyshev
        self._a = a
        self._b = b
        self._k = K  # Thường chọn K bằng 1
        self._eta = eta
        self.typicality = None
        super().__init__(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter, index=index, metric=metric)

    def __compute_gammas(self, centroids: np.ndarray) -> float:
        _d = distance_cdist(self.local_data, centroids)
        numerator = np.sum((self.membership ** self._m) * (_d ** 2), axis=0)
        denominator = np.sum((self.membership ** self._m), axis=0)
        return self._k * (numerator / denominator)

    def _update_typicality(self, centroids: np.ndarray):
        _gammas = self.__compute_gammas(centroids=centroids)
        _d = distance_cdist(self.local_data, centroids)
        denominator = 1 + ((self._b / _gammas)[None, :] * (_d ** 2)) ** (1 / (self._eta - 1))
        return 1 / denominator

    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        _tk = self._a * (membership ** self._m) + self._b * (self.typicality ** self._eta)
        numerator = np.dot(_tk.T, data)
        denominator = np.sum(_tk, axis=0)
        return numerator / denominator[:, None]

    def fit(self, data: np.ndarray, init_v: np.ndarray = None, seed: int = 0):
        self.local_data = data
        _start_tm = time.time()
        self.centroids = self._init_centroid_random(data, seed) if init_v is None else init_v
        for step in range(self._max_iter):
            old_v = self.centroids.copy()
            self.membership = self.update_membership(data=data, centroids=old_v)
            self.typicality = self._update_typicality(centroids=old_v)
            self.centroids = self._update_centroids(data=self.local_data, membership=self.membership)
            if self.check_exit_by_centroids(old_v):
                break
        self.process_time = time.time() - _start_tm
        return self.membership, self.centroids, step + 1


if __name__ == '__main__':
    from ds.clustering.utility import round_float, extract_labels
    from ds.clustering.utility import fetch_data_from_local, TEST_CASES, LabelEncoder
    from ds.clustering.validity import davies_bouldin, partition_coefficient, partition_entropy, dunn, classification_entropy, silhouette, hypervolume, cs, separation, calinski_harabasz

    ROUND_FLOAT = 4
    MAX_ITER = 1000
    DATA_ID = 53
    EPSILON = 1e-6
    LABELED_RATIOS = 0.7
    SEED = 42
    M = 2
    A = 1
    B = 1
    ETA = 2
    n_space = 10
    # ============================================

    SPLIT = '\t'

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def print_info(title: str, X: np.ndarray, U: np.ndarray, V: np.ndarray, process_time: float, step: int = 0, split: str = SPLIT) -> str:
        labels = extract_labels(U)  # Giai mo
        # print("Gán nhãn từng điểm X vào các cụm V:", labels)
        # print("Số lần xuất hiện các phần tử trong mỗi cụm:", count_data_array(labels))
        kqdg = [
            title,
            wdvl(process_time),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(partition_entropy(U)),  # PE
            wdvl(classification_entropy(U)),  # CE
            wdvl(silhouette(X, labels)),  # SI
            wdvl(hypervolume(U, M)),  # FHV
            wdvl(cs(X, U, V, M)),  # CS
            wdvl(separation(X, U, V, M), n=0),  # S
            wdvl(calinski_harabasz(X, labels)),  # CH
        ]
        result = split.join(kqdg)
        print(result)
        return result

    # ============================================
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _start_time = time.time()
        _dt = fetch_data_from_local(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        n_cluster = _TEST['n_cluster']

        print('#PFCM ==========================')
        dlec = LabelEncoder()
        labels = dlec.fit_transform(_dt['Y'].flatten())

        X = _dt['X']
        y_true = labels

        titles = ['Alg', 'time', 'step', 'DI', 'DB', 'PC', 'PE', 'CE', 'CH', 'SI', 'FHV', 'CS', 'S']
        print(SPLIT.join(titles))
        print('---------------------')
        # -----------------------
        fcmu = Dfcm(n_clusters=n_cluster, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        fcmu.fit(data=X,
                 seed=SEED)
        print_info(title='FCMU', X=X, U=fcmu.membership, V=fcmu.centroids, process_time=fcmu.process_time, step=fcmu.step)
        # # -----------------------
        pfcm = Dpfcm(n_clusters=n_cluster, m=M, epsilon=EPSILON, max_iter=MAX_ITER, a=A, b=B, eta=ETA)
        U, V, step = pfcm.fit(data=X,
                              init_v=np.array([[5.8, 2.7, 5.1, 1.9], [5.0, 2.0, 3.5, 1.0], [5.10, 3.50, 1.40, 0.30]]),
                              seed=SEED)
        print_info(title='PFCM', X=X, U=U, V=V, process_time=pfcm.process_time, step=step)

        print('V PFCM\n', V)
