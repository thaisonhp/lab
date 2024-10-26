from sklearn.utils import shuffle
import os
import re
import json
import numpy as np
import pandas as pd
from urllib import request, parse, error
import certifi
import ssl


COLORS = ['Blue', 'Orange', 'Green', 'Red', 'Cyan', 'Yellow', 'Purple', 'Pink', 'Brown', 'Black', 'Gray', 'Beige', 'Turquoise', 'Silver', 'Gold']
TEST_CASES = {
    14: {
        'name': 'BreastCancer',
        'n_cluster': 2,
        'test_points': ['30-39', 'premeno', '30-34', '0-2', 'no', 3, 'left', 'left_low', 'no']
    },
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    80: {
        'name': 'Digits',
        'n_cluster': 10,
        'test_points': [0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11, 2, 0, 0, 5, 16, 3, 0, 5, 7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0, 0, 0, 0, 6, 14, 7, 1, 0, 0]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 3,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    236: {
        'name': 'Seeds',
        'n_cluster': 3,
        'test_points': [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


# Mã hóa nhãn
class LabelEncoder:
    def __init__(self):
        self.index_to_label = {}
        self.unique_labels = None

    @property
    def classes_(self) -> np.ndarray:
        return self.unique_labels

    def fit_transform(self, labels) -> np.ndarray:
        self.unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return np.array([label_to_index[label] for label in labels])

    def inverse_transform(self, indices) -> np.ndarray:
        return np.array([self.index_to_label[index] for index in indices])


# =======================================
def numpy_load(filepath: str) -> np.ndarray:
    return np.loadtxt(filepath)


def numpy_save(filepath: str, data: np.ndarray, fmt: str = '%.18e', delimiter: str = ' '):
    np.savetxt(filepath, data, fmt=fmt, delimiter=delimiter)


# Ánh xạ mã mầu cho vẽ ảnh sau phân cụm dựa trên tâm cụm và bảng mầu của ảnh đã vẽ chuẩn
def map_centroids_to_colors(new_centroids: np.ndarray, centroids_path: str, color_path: str) -> np.ndarray:
    _centroids_standard = numpy_load(centroids_path)
    # print('centroids_standard', _centroids_standard.shape)
    # print(_centroids_standard)

    # print('new_centroids', new_centroids.shape)
    # print(new_centroids)
    # Tính khoảng cách giữa các centroids mới với centroids chuẩn
    distances = distance_cdist(new_centroids, _centroids_standard)

    # Khởi tạo mảng lưu kết quả ánh xạ giữa color_palette và new_color_palette
    C = new_centroids.shape[0]

    # Tạo mảng lưu index kết quả ánh xạ
    _kqaxs = np.full(C, -1)
    for _ in range(C):
        # Lấy ra vị trí của khoảng cách nhỏ nhất (min_index)
        new_centroid_index, standard_centroid_index = np.unravel_index(distances.argmin(), distances.shape)
        _kqaxs[new_centroid_index] = standard_centroid_index

        # Gán lại khoảng cách tới new_centroid_index là vô cực -> đã được xét
        distances[new_centroid_index, :] = np.inf

        # Gán lại khoảng cách tới điểm standard_centroid_index là vô cực -> đã được xét
        distances[:, standard_centroid_index] = np.inf

    _color_palette = numpy_load(color_path)
    # print('color_palette', _color_palette)
    # print('kq', _color_palette[_kqaxs])
    return _color_palette[_kqaxs]


# =======================================
# make_semi_supervised
def random_negative_assignment(labels: np.ndarray, ratio: float = 0.3, seed: int = 0, val: float = -1) -> np.ndarray:
    # Tính số lượng nhãn cần thay thế bằng [val]
    n_labels = len(labels)
    n_remove = int(ratio * n_labels)

    if seed > 0:
        np.random.seed(seed)
    # Lấy chỉ số ngẫu nhiên của những nhãn cần thay thế
    indices_to_remove = np.random.choice(n_labels, n_remove, replace=False)

    # Sao chép nhãn để không thay đổi y_true gốc
    result = labels.copy()
    # Thay thế các nhãn tại các vị trí này bằng [val]
    result[indices_to_remove] = val
    return result


def random_negative_assignment_class(labels: np.ndarray, ratio_per_class: list, seed: int = 0, val: float = -1) -> np.ndarray:
    if seed > 0:
        np.random.seed(seed)
    unique_labels = np.unique(labels)  # Các lớp có trong nhãn
    new_labels = labels.copy()  # Tạo một bản sao của nhãn ban đầu để thay đổi

    # Lặp qua từng lớp và thay thế nhãn theo tỷ lệ chỉ định
    for i, lbl in enumerate(unique_labels):
        class_indices = np.where(labels == lbl)[0]  # Các chỉ số của lớp hiện tại
        if i < len(ratio_per_class):
            ratio = ratio_per_class[i] / 100  # Chuyển tỷ lệ thành giá trị từ 0 đến 1
            num_to_replace = int(len(class_indices) * ratio)  # Số lượng nhãn sẽ bị thay thế
            indices_to_replace = np.random.choice(class_indices, size=num_to_replace, replace=False)
            new_labels[indices_to_replace] = val
    return new_labels


def random_negative_assignment_full_cluster(labels: np.ndarray, ratio: float = 0.3, seed: int = 0, val: float = -1) -> np.ndarray:
    llb = len(np.unique(labels))
    while True:
        y_lble = random_negative_assignment(labels=labels, ratio=ratio, seed=seed, val=val)
        if len(np.unique(y_lble)) == llb + 1:
            return y_lble


def split_data_for_semi_supervised_learning(data: np.ndarray, labels: np.ndarray, n_sites: int = 3, ratio: float = 0.3, seed: int = 0, val: float = -1, full_cluster: bool = False) -> list:
    result = []
    datas = np.array_split(data, n_sites)
    labeled = np.array_split(labels, n_sites)
    if not full_cluster:
        for i, data in enumerate(datas):
            y_true = labeled[i]
            y_lble = random_negative_assignment(labels=y_true, ratio=ratio, seed=seed, val=val)
            result.append({'X': data, 'Y': y_lble, 'T': y_true})
    else:
        for i, data in enumerate(datas):
            y_true = labeled[i]
            y_lble = random_negative_assignment_full_cluster(labels=y_true, ratio=ratio, seed=seed, val=val)
            result.append({'X': data, 'Y': y_lble, 'T': y_true})
    return result


def split_data_by_labels(data: np.ndarray, labels: np.ndarray, n_sites: int = 3, seed: int = 42) -> list:
    """
    Chia dữ liệu thành [n_sites] datasite sao cho mỗi datasite đều có đủ dữ liệu thuộc tất cả các cụm.

    Args:
        data: Dữ liệu đầu vào với kích thước (N, D), trong đó N là số điểm dữ liệu, D là số đặc trưng.
        labels: Nhãn tương ứng của dữ liệu, kích thước (N,).
        n_sites: Số lượng datasites cần chia.

    Returns:
        datasites (list): Danh sách chứa [n_sites] datasites, mỗi datasite là một dictionary với các keys 
                            'X' chứa điểm dữ liệu và 'Y' là nhãn tương ứng.
    """
    from sklearn.utils import shuffle

    unique_labels = np.unique(labels)  # Các nhãn duy nhất (các cụm)
    result = [{'X': [], 'Y': []} for _ in range(n_sites)]  # Khởi tạo danh sách các datasites

    # Lặp qua từng nhãn để chia đều dữ liệu vào các datasites
    for label in unique_labels:
        # Lấy các điểm dữ liệu thuộc cụm hiện tại
        label_data = data[labels == label]
        label_labels = labels[labels == label]

        # Shuffle các dữ liệu này để đảm bảo phân phối ngẫu nhiên
        label_data, label_labels = shuffle(label_data, label_labels, random_state=seed)

        # Chia đều dữ liệu vào các datasites
        split_data = np.array_split(label_data, n_sites)
        split_labels = np.array_split(label_labels, n_sites)

        # Phân phối dữ liệu vào từng datasite
        for i in range(n_sites):
            result[i]['X'].append(split_data[i])
            result[i]['Y'].append(split_labels[i])

    # Ghép lại danh sách dữ liệu và nhãn cho mỗi datasite
    for i in range(n_sites):
        result[i]['X'] = np.vstack(result[i]['X'])  # Ghép các mảng data lại
        result[i]['Y'] = np.concatenate(result[i]['Y'])  # Ghép các mảng labels lại

    return result


def name_slug(text: str, delim: str = '-') -> str:
    __punct_re = re.compile(r'[\t !’"“”#@$%&~\'()*\+:;\-/<=>?\[\\\]^_`{|},.]+')
    if text:
        from unidecode import unidecode
        result = [unidecode(word) for word in __punct_re.split(text.lower()) if word]
        result = [rs if rs != delim and rs.isalnum() else '' for rs in result]
        return re.sub(r'\s+', delim, delim.join(result).strip())


# Làm tròn số
def round_float(number: float, n: int = 3) -> float:
    if n == 0:
        return int(number)
    return round(number, n)


# Ma trận độ thuộc ra nhãn (giải mờ)
def extract_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1)


# Chia các điểm vào các cụm
def extract_clusters(data: np.ndarray, labels: np.ndarray, n_cluster: int = 0) -> list:
    if n_cluster == 0:
        n_cluster = np.unique(labels)
    return [data[labels == i] for i in range(n_cluster)]


# Chuẩn Euclidean của một vector đo lường độ dài của vector
# là căn bậc hai của tổng bình phương các phần tử của vector đó.
# d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
def norm_distances(A: np.ndarray, B: np.ndarray, axis: int = None) -> float:
    # np.sqrt(np.sum((np.asarray(A) - np.asarray(B)) ** 2))
    # np.sum(np.abs(np.array(A) - np.array(B)))
    return np.linalg.norm(A - B, axis=axis)


def minus_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X[:, np.newaxis, :] - Y[np.newaxis, :, :]


def distance_euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distance = minus_distances(X, Y)
    return np.sqrt(np.sum(distance ** 2, axis=2))


def distance_chebyshev(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distance = minus_distances(X, Y)
    return np.max(np.abs(distance), axis=2)


# Ma trận khoảng cách Euclide giữa các điểm trong 2 tập hợp dữ liệu
def distance_cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    # return distance_euclidean(X,Y) if metric=='euclidean' else distance_chebyshev(X,Y)
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)


# Khoảng cách của 2 cặp điểm trong một ma trận
def distance_pdist(data: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    from scipy.spatial.distance import pdist
    return pdist(data, metric=metric)


# lấy giá trị lớn nhất để tránh lỗi chia cho 0
def not_division_by_zero(data: np.ndarray):
    return np.fmax(data, np.finfo(np.float64).eps)


# Chuẩn hóa mỗi hàng của ma trận sao cho tổng của mỗi hàng bằng 1.
# \mathbf{x}_{norm} = \frac{\mathbf{x}}{\sum_{i=1}^m \mathbf{x}_{i,:}}
def standardize_rows(data: np.ndarray) -> np.ndarray:
    # Ma trận tổng của mỗi cột (cùng số chiều)
    _sum = np.sum(data, axis=0, keepdims=1)
    # Chia từng phần tử của ma trận cho tổng tương ứng của cột đó.
    return data / _sum


# Đếm số lần xuất hiện của từng phần tử trong 1 mảng
def count_data_array(data: np.ndarray) -> dict:
    unique_elements, counts = np.unique(data, return_counts=True)
    return {int(element): int(count) for element, count in zip(unique_elements, counts)}


def load_dataset(data: dict, file_csv: str = '', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    # label_name = data['data']['target_col']
    print('UCI uci_id=', data['data']['uci_id'], data['data']['name'])  # Mã + Tên bộ dữ liệu
    # print('data abstract=', data['data']['abstract'])  # Tên bộ dữ liệu
    # print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    # print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    # print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    metadata = data['data']
    # colnames = ['Area', 'Perimeter']
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    # print('data top', df.head())  # Hiển thị một số dòng dữ liệu
    # Trích xuất ma trận đặc trưng X (loại trừ nhãn lớp)
    return {'data': data['data'], 'ALL': df.iloc[:, :].values, 'X': df.iloc[:, :-1].values, 'Y': df.iloc[:, -1:].values}


# Lấy dữ liệu từ ổ cứng
def fetch_data_from_local(name_or_id=53, folder: str = '/home/dll/ncs/clustering/dataset', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    if isinstance(name_or_id, str):
        name = name_or_id
    else:
        name = TEST_CASES[name_or_id]['name']
    _folder = os.path.join(folder, name)
    fileio = os.path.join(_folder, 'api.json')
    if not os.path.isfile(fileio):
        print(f'File {fileio} not found!')
    with open(fileio, 'r') as cr:
        response = cr.read()
    return load_dataset(json.loads(response),
                        file_csv=os.path.join(_folder, 'data.csv'),
                        header=header, index_col=index_col, usecols=usecols, nrows=nrows)


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53, header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset'
    if isinstance(name_or_id, str):
        api_url += '?name=' + parse.quote(name_or_id)
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        _rs = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        response = _rs.read()
        _rs.close()
        return load_dataset(json.loads(response),
                            header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')


def draw_matplot(title: str, C: int, data: np.ndarray, labels: np.ndarray, centroids: np.ndarray, x_label: str, y_label: str, save2img: str = ''):
    import matplotlib.pyplot as plt
    plt.subplots()
    for i in range(C):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=COLORS[i], label=f'Cum {i+1}')

    plt.scatter(centroids[0], centroids[1], color='red', marker='x', label='Tam cum')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    if save2img:
        plt.savefig(save2img)


def show_matplot():
    import matplotlib.pyplot as plt
    plt.show()


# Ma trận khoảng cách Euclide giữa các điểm trong 2 tập hợp dữ liệu
def euclidean_cdist(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # _df = euclidean_distance(XA,XB)
    # return np.sqrt(_df)
    from scipy.spatial.distance import cdist
    return cdist(XA, XB)


def split_data_into_datasites(data: np.ndarray, labels: np.ndarray, P: int) -> list[dict]:
    """
    Chia dữ liệu thành P datasite sao cho mỗi datasite đều có đủ dữ liệu thuộc tất cả các cụm.

    Args:
        data (np.ndarray): Dữ liệu đầu vào với kích thước (N, D), trong đó N là số điểm dữ liệu, D là số đặc trưng.
        labels (np.ndarray): Nhãn tương ứng của dữ liệu, kích thước (N,).
        P (int): Số lượng datasites cần chia.

    Returns:
        datasites (list): Danh sách chứa P datasites, mỗi datasite là một dictionary với các keys 'data' và 'labels'.
                          'data' chứa các điểm dữ liệu, 'labels' chứa nhãn tương ứng.
    """
    unique_labels = np.unique(labels)  # Các nhãn duy nhất (các cụm)
    datasites = [{'data': [], 'labels': []} for _ in range(P)]  # Khởi tạo danh sách các datasites

    # Lặp qua từng nhãn để chia đều dữ liệu vào các datasites
    for label in unique_labels:
        # Lấy các điểm dữ liệu thuộc cụm hiện tại
        label_data = data[labels == label]
        label_labels = labels[labels == label]

        # Shuffle các dữ liệu này để đảm bảo phân phối ngẫu nhiên
        label_data, label_labels = shuffle(label_data, label_labels, random_state=42)

        # Chia đều dữ liệu vào các datasites
        split_data = np.array_split(label_data, P)
        split_labels = np.array_split(label_labels, P)

        # Phân phối dữ liệu vào từng datasite
        for i in range(P):
            datasites[i]['data'].append(split_data[i])
            datasites[i]['labels'].append(split_labels[i])

    # Ghép lại danh sách dữ liệu và nhãn cho mỗi datasite
    for i in range(P):
        datasites[i]['data'] = np.vstack(datasites[i]['data'])  # Ghép các mảng data lại
        datasites[i]['labels'] = np.concatenate(datasites[i]['labels'])  # Ghép các mảng labels lại

    return datasites
