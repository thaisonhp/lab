# Indices of Cluster Validity
import numpy as np
import math
from f1.clustering.utility import norm_distances, distance_pdist, distance_cdist


# DI fast
def dunn_fast(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics.pairwise import euclidean_distances

    def __delta_fast(ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]
        return np.min(values)

    def __big_delta_fast(ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        # values = values [np.nonzero(values)]
        return np.max(values)
    # -----------------------------------
    distances = euclidean_distances(data)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = __delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        big_deltas[k] = __big_delta_fast((labels == ks[k]), distances)
    return np.min(deltas)/np.max(big_deltas)


# DI
def dunn(data: np.ndarray, labels: np.ndarray) -> float:
    C = len(np.unique(labels))
    cluster_points = [data[labels == i] for i in range(C)]
    cluster_centers = np.array([np.mean(points, axis=0) for points in cluster_points])
    # Tính khoảng cách nhỏ nhất giữa các tâm cụm
    min_cluster_distance = np.inf
    from itertools import combinations
    for i, j in combinations(range(C), 2):
        dist = norm_distances(cluster_centers[i], cluster_centers[j])
        min_cluster_distance = min(min_cluster_distance, dist)
    # Tính đường kính lớn nhất của các cụm
    max_cluster_diameter = 0
    for points in cluster_points:
        if len(points) > 1:  # Cụm phải có ít nhất 2 điểm để tính đường kính
            distances = norm_distances(points[:, np.newaxis], points, axis=2)
            cluster_diameter = np.max(distances)
            max_cluster_diameter = max(max_cluster_diameter, cluster_diameter)
    # Tránh chia cho 0
    if max_cluster_diameter == 0:
        return np.inf
    return min_cluster_distance / max_cluster_diameter
    # =============================================
    # from scipy.spatial.distance import cdist, pdist, squareform
    # distances = pdist(data)
    # dist_matrix = squareform(distances)

    # labels_unique = np.unique(labels)
    # n_clusters = len(labels_unique)

    # min_inter_cluster_distance = np.inf
    # max_intra_cluster_distance = 0

    # for k in range(n_clusters):
    #     cluster_k = data[labels == k]

    #     # Tính khoảng cách lớn nhất trong cụm
    #     if len(cluster_k) > 1:
    #         max_intra_cluster_distance = max(
    #             max_intra_cluster_distance,
    #             np.max(pdist(cluster_k))
    #         )

    #     # Tính khoảng cách nhỏ nhất giữa các cụm
    #     for l in range(k + 1, n_clusters):
    #         cluster_l = data[labels == l]
    #         min_dist = np.min(cdist(cluster_k, cluster_l).flatten())
    #         min_inter_cluster_distance = min(min_inter_cluster_distance, min_dist)

    # if max_intra_cluster_distance == 0:
    #     return np.inf
    # return min_inter_cluster_distance / max_intra_cluster_distance


# DB
def davies_bouldin(data: np.ndarray, labels: np.ndarray) -> float:
    # C = len(np.unique(labels))
    # cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(C)])

    # # Tính độ lệch chuẩn cho mỗi cụm
    # # dispersions = np.zeros(n_clusters)
    # dispersions = [np.mean(norm_distances(data[labels == i], cluster_centers[i], axis=1)) for i in range(C)]

    # result = 0
    # for i in range(C):
    #     max_ratio = 0
    #     for j in range(C):
    #         if i != j:
    #             ratio = (dispersions[i] + dispersions[j]) / norm_distances(cluster_centers[i], cluster_centers[j])
    #             max_ratio = max(max_ratio, ratio)
    #     result += max_ratio
    # return result / C
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(data, labels)


# PCI fuzzy
def partition_coefficient(membership: np.ndarray) -> float:
    N, C = membership.shape
    _pc = np.sum(np.square(membership)) / N  # PC fuzzy
    _1dc = 1/C
    return (_pc - _1dc) / (1 - _1dc)


# CE fuzzy
def classification_entropy(membership: np.ndarray, a: float = np.e) -> float:
    """
    CE đo lường mức độ không chắc chắn trong việc gán điểm vào các cụm, giá trị càng thấp, độ hợp lệ
    của phân cụm càng tốt. CE thường kết hợp với PC, một phân cụm tốt thường có PC cao và CE thấp,
    0 ≤ 1 − P C ≤ CE
    """
    N = membership.shape[0]

    # Tránh log(0) bằng cách thêm một epsilon nhỏ cho tất cả các phần tử
    epsilon = np.finfo(float).eps
    membership = np.clip(membership, epsilon, 1)

    # Tính tỉ lệ phần trăm điểm dữ liệu thuộc về mỗi cụm
    log_u = np.log(membership) / np.log(a)  # Chuyển đổi cơ số logarit
    return -np.sum(membership * log_u) / N


# PE fuzzy
def partition_entropy(membership: np.ndarray) -> float:
    """
    Tính chỉ số Partition Entropy index

    Parameters
    ----------
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Partition Entropy index
    """
    return classification_entropy(membership=membership, a=np.e)


def purity_score(membership: np.ndarray) -> float:
    """
    Tính chỉ số Purity index

    Parameters
    ----------
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Purity index
    """
    return np.mean([np.max(membership[i]) for i in range(len(membership))])


# S fuzzy
def separation(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    _N, C = membership.shape
    _ut = membership.T
    numerator = 0
    for i in range(C):
        diff = data - centroids[i]
        squared_diff = np.sum(diff**2, axis=1)
        numerator += np.sum((_ut[i] ** m) * squared_diff)
    center_dists = np.sum((centroids[:, np.newaxis] - centroids) ** 2, axis=2)
    np.fill_diagonal(center_dists, np.inf)
    min_center_dist = np.min(center_dists)
    return numerator / min_center_dist


# CH
def calinski_harabasz(data: np.ndarray, labels: np.ndarray) -> float:
    # N = len(data)
    # C = len(np.unique(labels))

    # # Tính tổng phương sai
    # overall_mean = np.mean(data, axis=0)
    # overall_var = np.sum((data - overall_mean) ** 2)

    # # Tính phương sai giữa các cụm
    # between_var = np.sum([len(np.where(labels == i)[0]) *
    #                       np.sum((np.mean(data[labels == i], axis=0) - overall_mean) ** 2)
    #                       for i in range(C)])

    # # Tính phương sai trong cụm
    # within_var = overall_var - between_var
    # if N == C or C == 1:
    #     return 0
    # return (between_var / (C - 1)) / (within_var / (N - C))
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(data, labels)


# SI
def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    # N = len(data)
    # silhouette_vals = np.zeros(N)
    # for i in range(N):
    #     a_i = 0
    #     b_i = np.inf
    #     for j in range(N):
    #         if i != j:
    #             distance = np.sqrt(np.sum((data[i] - data[j])**2))
    #             if labels[i] == labels[j]:
    #                 a_i += distance
    #             else:
    #                 b_i = min(b_i, distance)

    #     if np.sum(labels == labels[i]) > 1:
    #         a_i /= (np.sum(labels == labels[i]) - 1)
    #     else:
    #         a_i = 0
    #     silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    # return np.mean(silhouette_vals)
    from sklearn.metrics import silhouette_score
    return silhouette_score(data, labels)


# AC
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    # total_samples = len(y_true)
    # return correct_predictions / total_samples
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def tp_fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # Tính TP, FP, FN
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    return tp, fp, fn


# F1
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:  # binary|weighted
    # tp, fp, fn = tp_fp_fn(y_true, y_pred)
    # # Tính precision và recall
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # total = precision + recall
    # return 2 * (precision * recall) / total if total > 0 else 0
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)


# precision_score
def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:  # binary|weighted
    # tp, fp, fn = tp_fp_fn(y_true, y_pred)
    # return tp / (tp + fp) if (tp + fp) > 0 else 0
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average=average)


# recall_score
def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:  # binary|weighted
    # tp, fp, fn = tp_fp_fn(y_true, y_pred)
    # return tp / (tp + fn) if (tp + fn) > 0 else 0
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average=average)


# FHV fuzzy
def hypervolume(membership: np.ndarray, m: float = 2) -> float:
    C = membership.shape[1]
    result = 0
    for i in range(C):
        cluster_u = membership[:, i]
        n_i = np.sum(cluster_u > 0)
        if n_i > 0:
            result += np.sum(cluster_u ** m) / n_i
    return result


# CS fuzzy
def cs(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    N, C = membership.shape
    numerator = 0
    for i in range(C):
        numerator += np.sum((membership[:, i]**m)[:, np.newaxis] *
                            np.sum((data - centroids[i])**2, axis=1)[:, np.newaxis])
    min_center_dist = np.min([np.sum((centroids[i] - centroids[j])**2)
                              for i in range(C)
                              for j in range(i+1, C)])
    return numerator / (N * min_center_dist)


# XB fuzzy
def Xie_Benie(data: np.ndarray, centroids: np.ndarray, membership: np.ndarray) -> float:
    """
    Tính chỉ số Xie-Benie index

    Parameters
    ----------
    data: dữ liệu chưa được phân cụm.
    clusters: Ma trận các điểm đã được chia về cụm tương ứng bằng cách giải mờ.
    centroids: Ma trận tâm cụm 
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Xie-Benie index
    """
    _N, C = membership.shape
    labels = np.argmax(membership, axis=1)
    clusters = [data[labels == i] for i in range(C)]

    from sklearn.metrics import pairwise_distances
    S_iq = np.asanyarray([np.mean([np.linalg.norm(point - centroids[i]) for point in cluster]) for i, cluster in enumerate(clusters)])
    tu = np.sum(np.square(membership) * np.square(S_iq))
    distance = pairwise_distances(centroids)
    distance[distance == 0] = math.inf
    mau = len(data) * np.min(np.square(distance))
    return tu / mau


# Trung bình khoảng cách các tâm
def mean_distance_cluster(centroids: np.ndarray) -> float:
    # C = centroids.shape[0]
    # if C < 2:
    #     return 0.0

    # # Tính ma trận khoảng cách giữa tất cả các cặp tâm cụm
    # diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    # distances = np.sqrt(np.sum(diff**2, axis=-1))

    # # Chỉ lấy nửa trên của ma trận đối xứng (không tính của chính nó)
    # mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    # distances = distances[mask]
    # # # Tính khoảng cách Euclid giữa tất cả các cặp tâm cụm
    # # distances = np.zeros((C, C))
    # # for i in range(C):
    # #     for j in range(i + 1, C):
    # #         distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    # # # Chỉ lấy nửa trên của ma trận đối xứng (không tính của chính nó)
    # # distances = distances[np.triu_indices(C, k=1)]

    # return np.mean(distances)  # Tính tổng và trung bình khoảng cách
    # Tính khoảng cách giữa tất cả các cặp tâm cụm
    distances = distance_pdist(centroids)
    return np.mean(distances)


# Trung bình tổng khoảng cách điểm tới tâm
def mean_distance_point_cluster(data: np.ndarray, centroids: np.ndarray) -> float:
    # # Tính ma trận khoảng cách giữa tất cả các điểm dữ liệu và tất cả các tâm cụm
    # diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    # distances = np.sqrt(np.sum(diff**2, axis=-1))
    # # Tính tổng khoảng cách của mỗi điểm tới tất cả các tâm cụm
    # total_distances = np.sum(distances, axis=1)
    # # Tính giá trị trung bình của các tổng khoảng cách
    # return np.mean(total_distances)
    distances = distance_cdist(data, centroids)
    # Tính tổng khoảng cách của mỗi điểm tới tất cả các tâm cụm
    total_distances = np.sum(distances, axis=1)
    # Tính giá trị trung bình của các tổng khoảng cách
    return np.mean(total_distances)
