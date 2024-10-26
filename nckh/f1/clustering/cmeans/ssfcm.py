# HoangNX update 18/10/2024
import numpy as np
from f1.clustering.cmeans.fcm import Dfcm


class Dssfcm(Dfcm):
    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000, index: int = 0, metric: str = 'euclidean'):
        self.__no_label = -1
        super().__init__(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter, index=index, metric=metric)
        self.labeled = None
        self.membership_bar = None

    @property
    def ratio_labeled(self) -> float:
        if not self.labeled is None:
            mask = self.labeled > -1
            return np.sum(mask) / len(self.local_data)
        return 0

    # Khởi tạo điểm dữ liệu có nhãn (bán giám sát)
    def init_membership_bar(self, n_samples: int):
        self.membership_bar = np.zeros((n_samples, self._n_clusters))
        if self.labeled is not None:
            for i, label in enumerate(self.labeled):
                if label != self.__no_label:
                    self.membership_bar[i, label] = 1

    # Cập nhật ma trận tâm cụm
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        _um = (membership - self.membership_bar) ** self._m  # (N, C)
        numerator = np.dot(_um.T, data)  # (C, N) x (N, D) = (C, D)
        denominator = _um.sum(axis=0)[:, np.newaxis]  # (C, 1)  |   [:, np.newaxis] <=> .reshape(C,1)
        return numerator / self._division_by_zero(denominator)

    # Cập nhật ma trận độ thuộc
    def calculate_membership(self, distances: np.ndarray) -> np.ndarray:
        sum_membar = np.sum(self.membership_bar, axis=1, keepdims=True)
        numerator = 1 - sum_membar
        _d = distances[:, :, None] * ((1 / self._division_by_zero(distances))[:, None, :])
        power = 2 / (self._m - 1)
        denominator = (_d ** power).sum(axis=2)
        return self.membership_bar + numerator / denominator

    def fit(self, data: np.ndarray, labeled: np.ndarray, seed: int = 0, no_label: int = -1) -> tuple:
        self.__no_label = no_label
        self.labeled = labeled
        _n_samples = len(data)
        self.init_membership_bar(n_samples=_n_samples)
        return super().fit(data=data, seed=seed)


if __name__ == "__main__":
    import time
    from f1.clustering.utility import count_data_array, extract_labels, round_float
    from f1.clustering.utility import fetch_data_from_local, TEST_CASES, LabelEncoder, random_negative_assignment
    from f1.clustering.validity import partition_coefficient, calinski_harabasz, dunn, davies_bouldin
    from f1.clustering.validity import partition_entropy, Xie_Benie, f1_score, accuracy_score, Xie_Benie
    from f1.clustering.validity import silhouette, hypervolume

    ROUND_FLOAT = 3
    MAX_ITER = 1000
    DATA_ID = 602  # 14: Breast Cancer, 53: Iris, 80: Digits, 109: Wine, 236: Seeds, 602: DryBean
    EPSILON = 1e-5
    LABELED_RATIOS = 0.7
    SEED = 42
    M = 2
    # SPLIT = '&'
    SPLIT = '\t'
    NO_LABELED = -1

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def print_info(title: str, X: np.ndarray, U: np.ndarray, V: np.ndarray, y_true: np.ndarray = None, process_time: float = 0, step: int = 0, split: str = SPLIT) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            title,
            wdvl(process_time),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(partition_entropy(U)),  # PE
            wdvl(silhouette(X, labels)),  # SI
            wdvl(hypervolume(U, M)),  # FHV
            wdvl(calinski_harabasz(X, labels)),  # CH
            wdvl(Xie_Benie(X, V, U)),  # XB
            '0' if y_true is None else wdvl(f1_score(y_true, labels)),
            '0' if y_true is None else wdvl(accuracy_score(y_true, labels))
        ]
        result = split.join(kqdg)
        print(result)
        return result

    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _start_time = time.time()
        _dt = fetch_data_from_local(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        n_cluster = _TEST['n_cluster']  # Số lượng cụm

        dlec = LabelEncoder()
        labels = dlec.fit_transform(_dt['Y'].flatten())

        X = _dt['X']
        Y = random_negative_assignment(labels=labels, ratio=1 - LABELED_RATIOS, seed=SEED, val=NO_LABELED)
        print('X', X.shape, X[0], Y[0])
        print('Y', Y.shape, count_data_array(Y))
        # ----------------------------------------
        titles = ['Alg', 'time', 'step', 'DI', 'DB', 'PC', 'PE', 'CH', 'SI', 'FHV', 'XB', 'F1', 'AC']
        print(SPLIT.join(titles))
        print('---------------------')
        # ----------------------------------------
        fcm = Dfcm(n_clusters=n_cluster, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        fcm.fit(data=X, seed=SEED)
        print_info(title='Dfcm RD', X=fcm.local_data, U=fcm.membership, V=fcm.centroids, process_time=fcm.process_time, step=fcm.step)

        ssfcm = Dssfcm(n_clusters=n_cluster, epsilon=EPSILON, max_iter=MAX_ITER)
        ssfcm.fit(data=X, labeled=Y, seed=SEED, no_label=NO_LABELED)
        print_info(title='Dssfcm', X=ssfcm.local_data, U=ssfcm.membership, V=ssfcm.centroids, y_true=labels, process_time=ssfcm.process_time, step=ssfcm.step)
        print('ratio_labeled=', round_float(ssfcm.ratio_labeled))
        # ----------------------------------------
        print("Số lần xuất hiện các phần tử trong mỗi cụm SSFCM:", count_data_array(ssfcm.extract_labels))
