import time
import numpy as np


if __name__ == '__main__':


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
        _dt = fetch_data_from_local(DATA_ID, folder='/home/dll/ncs/clustering/dataset')
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        n_cluster = _TEST['n_cluster']

        print('#PFCM ==========================')

        X = _dt['X']

        titles = ['Alg', 'time', 'step', 'DI', 'DB', 'PC', 'PE', 'CE', 'CH', 'SI', 'FHV', 'CS', 'S']
        print(SPLIT.join(titles))
        print('---------------------')
        # -----------------------
        fcmu = Dfcm(n_clusters=n_cluster, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        fcmu.fit(data=X,
                 seed=SEED)
        print_info(title='FCMU', X=X, U=fcmu.membership, V=fcmu.centroids, process_time=fcmu.process_time, step=fcmu.step)
        # -----------------------
        fcmv = Dfcm(n_clusters=n_cluster, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        fcmv.fit(data=X,
                 init_v=np.array([[5.8, 2.7, 5.1, 1.9], [5.0, 2.0, 3.5, 1.0], [5.10, 3.50, 1.40, 0.30]]),
                 with_u=False,
                 seed=SEED)
        print_info(title='FCMV', X=X, U=fcmv.membership, V=fcmv.centroids, process_time=fcmv.process_time, step=fcmv.step)
        # -----------------------
        pfcm = Dpfcm(n_clusters=n_cluster, m=M, epsilon=EPSILON, max_iter=MAX_ITER, a=A, b=B, eta=ETA)
        U, V, step = pfcm.fit(data=X,
                              init_v=np.array([[5.8, 2.7, 5.1, 1.9], [5.0, 2.0, 3.5, 1.0], [5.10, 3.50, 1.40, 0.30]]),
                              seed=SEED)
        print_info(title='PFCM', X=X, U=U, V=V, process_time=pfcm.process_time, step=step)

        print('V FCMU\n', fcmu.centroids)
        print('V FCMV\n', fcmv.centroids)
        print('V PFCM\n', V)
