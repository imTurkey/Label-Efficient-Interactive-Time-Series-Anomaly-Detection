from collections import deque
import math
import numpy as np
import collections
# from luminol.anomaly_detector import AnomalyDetector

ZERO = 0.00000001

def mean_std_normalize(p: list):
    p = [0 if _ is None else _ for _ in p]
    p = np.array(p)
    mean = p.mean()
    std = p.std()
    if std < ZERO:
        std = ZERO
    rs = (p - mean) / std
    return rs.tolist()

def ma_diff_ratio_features(values, windows):
    names = []
    features = []

    for w in windows:
        names.append('ma_' + str(w))
        tmp_feature = []
        tmp_sum = 0
        for i in range(len(values)):
            if i - w < -1:
                tmp_sum += values[i]
                tmp_feature.append(None)
            else:
                if i - w == -1:
                    tmp_sum += values[i]
                else:
                    tmp_sum += values[i] - values[i - w]
                tmp_feature.append(tmp_sum / w)
        features.append(tmp_feature)

        names.append('ma_diff_' + str(w))
        tmp_feature_1 = []
        for i in range(len(tmp_feature)):
            if tmp_feature[i] is None:
                tmp_feature_1.append(None)
            else:
                tmp_feature_1.append(values[i] - tmp_feature[i])
        features.append(tmp_feature_1)

        names.append('ma_ratio_' + str(w))
        tmp_feature_2 = []
        for i in range(len(tmp_feature)):
            if tmp_feature[i] is None:
                tmp_feature_2.append(None)
            else:
                safe_divisor = tmp_feature[i]
                if tmp_feature[i] < ZERO:
                    safe_divisor = ZERO
                tmp_feature_2.append(values[i] / safe_divisor)
        features.append(tmp_feature_2)

    return names, features


def ewma_diff_ratio_features(values, decays, windows):
    names = []
    features = []
    for w in windows:
        for d in decays:
            base = 0
            tmp = 1
            for i in range(w):
                base += tmp
                tmp *= (1 - d)

            names.append('ewma_' + str(w) + '_' + str(d))
            tmp_feature = []

            for i in range(len(values)):
                if i - w < -1:
                    tmp_feature.append(None)
                else:
                    factor = 1
                    numeraotr = 0
                    for j in range(w):
                        numeraotr += values[i - j] * factor
                        factor *= (1 - d)
                    tmp_feature.append(numeraotr / base)
            features.append(tmp_feature)

            names.append('ewma_diff_' + str(w) + '_' + str(d))
            tmp_feature_1 = []

            for i in range(len(tmp_feature)):
                if tmp_feature[i] is None:
                    tmp_feature_1.append(None)
                else:
                    tmp_feature_1.append(values[i] - tmp_feature[i])
            features.append(tmp_feature_1)

            names.append('ewma_ratio_' + str(w) + '_' + str(d))
            tmp_feature_2 = []

            for i in range(len(tmp_feature)):
                if tmp_feature[i] is None:
                    tmp_feature_2.append(None)
                else:
                    safe_divisor = tmp_feature[i]
                    if tmp_feature[i] < ZERO:
                        safe_divisor = ZERO
                    tmp_feature_2.append(values[i] / safe_divisor)
            features.append(tmp_feature_2)

    return names, features


def win_min_max_diff_std_quantile(values, windows):
    names = []
    features = []
    for w in windows:
        tmp_min = []
        tmp_deque = deque()
        for i in range(len(values)):
            while len(tmp_deque) > 0 and tmp_deque[-1][0] >= values[i]:
                tmp_deque.pop()
            tmp_deque.append((values[i], i))

            if i < w - 1:
                tmp_min.append(None)
            else:
                if tmp_deque[0][1] == i - w:
                    tmp_deque.popleft()
                tmp_min.append(tmp_deque[0][0])
        names.append('min_' + str(w))
        features.append(tmp_min)

        tmp_max = []
        tmp_deque = deque()
        for i in range(len(values)):
            while len(tmp_deque) > 0 and tmp_deque[-1][0] <= values[i]:
                tmp_deque.pop()
            tmp_deque.append((values[i], i))

            if i < w - 1:
                tmp_max.append(None)
            else:
                if tmp_deque[0][1] == i - w:
                    tmp_deque.popleft()
                tmp_max.append(tmp_deque[0][0])
        names.append('max_' + str(w))
        features.append(tmp_max)

        tmp_diff = []
        for i in range(len(tmp_min)):
            if tmp_min[i] is None:
                tmp_diff.append(None)
            else:
                tmp_diff.append(tmp_max[i] - tmp_min[i])
        names.append('min_max_diff_' + str(w))
        features.append(tmp_diff)

        ss, ss2 = 0, 0
        tmp_std = []
        for i in range(len(values)):
            ss += values[i]
            ss2 += values[i] * values[i]

            if i < w - 1:
                tmp_std.append(None)
            else:
                if i > w - 1:
                    ss -= values[i - w]
                    ss2 -= values[i - w] * values[i - w]

                tmp_std.append(math.sqrt(math.fabs((ss2 - ss * ss / w) / (w - 1))))

        names.append('std_' + str(w))
        features.append(tmp_std)

        tmp_quantile = []
        for i in range(len(values)):
            if i < w - 1:
                 tmp_quantile.append(None)
            else:
                tmp_q = 1
                for j in range(1, w):
                    if values[i - j] >= values[i]:
                        tmp_q += 1
                tmp_quantile.append(tmp_q / w)
        names.append('quantile_' + str(w))
        features.append(tmp_quantile)

    return names, features


def diff_ratio(values):
    names = []
    features = []

    tmp_diff = [None]
    for i in range(1, len(values)):
        tmp_diff.append(values[i] - values[i - 1])
    names.append('diff')
    features.append(tmp_diff)

    tmp_diff_ratio = [None]
    for i in range(1, len(values)):
        divisor = values[i - 1]
        if divisor <= ZERO:
            divisor = ZERO
        tmp_diff_ratio.append(tmp_diff[i] / divisor)
    names.append('diff_ratio')
    features.append(tmp_diff_ratio)

    return names, features


def log_minus_divid(values):
    names = []
    features = []

    values_t = [math.fabs(x) + 1.1 for x in values]

    tmp_log = []
    tmp_minus = []
    tmp_divid = []
    for i in range(len(values_t)):
        tmp_log.append(math.log2(values_t[i]))
        tmp_minus.append(values_t[i] - tmp_log[i])
        tmp_divid.append(values_t[i] / tmp_log[i])
    names.append('log')
    features.append(tmp_log)
    names.append('log_minus')
    features.append(tmp_minus)
    names.append('log_divid')
    features.append(tmp_divid)

    return names, features


def spectral_residual(values, win_size):

    def average_filter(a, n=3):
        res = np.cumsum(a, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n
        for i in range(1, n):
            res[i] /= (i + 1)
        return res

    def backadd_new(a):
        backaddnum = 5
        kkk = (a[-2] - a[-5])/3
        kk = (a[-2] - a[-4])/2
        k = a[-2] - a[-3]
        kkkk = (a[-2] - a[-5])/4
        kkkkk = (a[-2] - a[-6])/5
        toadd = a[-5] + k + kk + kkk + kkkk + kkkkk
        a.append(toadd)
        for i in range(backaddnum-1):
            a.append(a[-1])
        return a

    length = len(values)

    detres = [None] * (win_size - 1)

    for pt in range(win_size - 1, length):

        head = pt + 1 - win_size

        tail = pt + 1

        wave = np.array(backadd_new(values[head:tail]))
        trans = np.fft.fft(wave)
        realnum = np.real(trans)
        comnum = np.imag(trans)
        mag = np.sqrt(realnum ** 2 + comnum ** 2)

        mag = np.clip(mag, ZERO, mag.max())

        spectral = np.exp(np.log(mag) - average_filter(np.log(mag)))
        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag
        wave = np.fft.ifft(trans)
        mag = np.sqrt(wave.real ** 2 + wave.imag ** 2)
        judgeavg = average_filter(mag, n=21)

        idx = win_size - 1
        safe_divisor = judgeavg[idx]
        if abs(safe_divisor) < ZERO:
            safe_divisor = ZERO
        detres.append(abs(mag[idx] - safe_divisor) / abs(safe_divisor))

    return ['sr'], [detres]


def median_diff_ratio_features(values, windows):
    names = []
    features = []

    for w in windows:
        names.append('median_' + str(w))
        tmp_feature = []
        for i in range(len(values)):
            if i - w < -1:
                tmp_feature.append(None)
            else:
                tmp_list = values[i - w + 1: i + 1]
                tmp_median = np.median(tmp_list)
                tmp_feature.append(tmp_median)
        features.append(tmp_feature)

        names.append('median_diff_' + str(w))
        tmp_feature_1 = []
        for i in range(len(tmp_feature)):
            if tmp_feature[i] is None:
                tmp_feature_1.append(None)
            else:
                tmp_feature_1.append(abs(values[i] - tmp_feature[i]))
        features.append(tmp_feature_1)

        names.append('median_ratio_' + str(w))
        tmp_feature_2 = []
        for i in range(len(tmp_feature)):
            if tmp_feature[i] is None:
                tmp_feature_2.append(None)
            else:
                divisor = tmp_feature[i]
                if abs(divisor) < ZERO:
                    divisor = ZERO
                tmp_feature_2.append(abs(values[i] / divisor))
        features.append(tmp_feature_2)

    return names, features


def fft_base_streaming(value, win):
    def complex_module(c):
        return math.sqrt(c.real * c.real + c.imag * c.imag)

    def filter_fre(fre, k_ratio=0.9):
        k = int(len(fre) * k_ratio)
        amp = [complex_module(_) for _ in fre]
        amp.sort(reverse=True)
        threshold = amp[k - 1]

        rs = []
        for i in range(len(fre)):
            if complex_module(fre[i]) >= threshold:
                rs.append(fre[i])
            else:
                rs.append(0)
        return rs

    def calculate_local_outlier(values, c=3):
        y = np.fft.fft(values).tolist()
        f2 = [_.real for _ in np.fft.ifft(filter_fre(y)).tolist()]

        values = np.array(values)
        # print(values.shape)
        f2 = np.array(f2)
        so = np.abs(f2 - values)
        mso = np.mean(so)

        s, sidx = [], []
        for i in range(values.shape[0]):
            if so[i] > mso:
                start = max(i - c, 0)
                end = min(i + c + 1, values.shape[0])
                nav = np.mean(values[start:end])
                s.append(values[i] - nav)
                sidx.append(i)

        if win - 1 not in sidx:
            return 0

        s = np.array(s)
        ms = np.mean(s)
        sds = np.std(s)

        idx = sidx.index(win - 1)
        sds = max(sds, ZERO)

        return abs(s[idx] - ms) / sds

    names = ['fft']

    pre = [0] * (win - 1)
    for i in range(win - 1, len(value)):
        start, end = i - win + 1, i + 1
        pre.append(calculate_local_outlier(value[start: end], c=10))
    rs = [pre]
    return names, rs


def luminol_streaming(timestamp, value):
    def list_sorted(lst):
        for i in range(1, len(lst)):
            if lst[i] >= lst[i - 1]:
                pass
            else:
                return False
        return True

    def most_frequent_interval(lst):
        lst = lst[:]
        if not list_sorted(lst):
            lst.sort()
        count = collections.Counter([lst[i] - lst[i - 1] for i in range(1, len(lst))])
        rs = list(count.items())
        rs.sort(key=lambda _: _[1], reverse=True)
        return rs[0][0]

    def timestamp_zero_start(timestamp):
        interval = most_frequent_interval(timestamp)
        min_ = min(timestamp)
        return [(_ - min_) // interval for _ in timestamp], min_, interval

    def call(timestamp, value):
        timestamp, min_, interval = timestamp_zero_start(timestamp)
        ts = {timestamp[i]: value[i] for i in range(len(value))}
        my_detector = AnomalyDetector(
            ts,
            algorithm_params={
                'precision': 2,
                'lag_window_size': 2,
                'future_window_size': 2,
                'chunk_size': 5
            },
            score_threshold=None,
            score_percent_threshold=0.95
        )
        anomaly = my_detector.get_anomalies()
        rs = {_: 0 for _ in timestamp}
        for interval in anomaly:
            for ts in range(interval.start_timestamp, interval.end_timestamp + 1):
                if ts in rs:
                    rs[ts] = 1
        rs = list(rs.items())
        rs.sort()
        return [_[1] for _ in rs]

    names = ['luminol']
    win_size = 128

    pre = [0] * (win_size - 1)
    for i in range(win_size - 1, len(timestamp)):
        start, end = i - win_size + 1, i + 1
        pre.append(call(timestamp[start: end], value[start: end])[-1])
    rs = [pre]

    return names, rs


def get_features(timestamp, values, with_sr):
    """
    :param timestamp:
    :param values:
    :return: list of features
    """
    names = []
    features = []

    # MA and MA_Diff
    tmp_names, tmp_features = ma_diff_ratio_features(values, windows=[10, 50, 100, 500, 1440])
    names += tmp_names
    features += tmp_features

    # EWMA and EWMA_Diff
    tmp_names, tmp_features = ewma_diff_ratio_features(values, decays=[0.1, 0.5], windows=[10, 50, 100, 500, 1440])
    names += tmp_names
    features += tmp_features

    # window min, max, min_max_diff, std, quantile
    tmp_names, tmp_features = win_min_max_diff_std_quantile(values, windows=[10, 50, 200, 500, 1440])
    names += tmp_names
    features += tmp_features

    # diff and ratio of change
    tmp_names, tmp_features = diff_ratio(values)
    names += tmp_names
    features += tmp_features

    # log, minus, divid
    tmp_names, tmp_features = log_minus_divid(values)
    names += tmp_names
    features += tmp_features

    # sr
    if with_sr:
        tmp_names, tmp_features = spectral_residual(values, win_size=1440)
        names += tmp_names
        features += tmp_features

    # we add the mean diff ratio, fft and luminol as extra features but see no improvement.
    # median diff ratio
    # tmp_names, tmp_features = median_diff_ratio_features(values, windows=[10, 100])
    # names += tmp_names
    # features += tmp_features

    # # fft
    # tmp_names, tmp_features = fft_base_streaming(values, win=128)
    # names += tmp_names
    # features += tmp_features

    # # luminol
    # tmp_names, tmp_features = luminol_streaming(timestamp, values)
    # names += tmp_names
    # features += tmp_features

    # replace value None with 0 and normalize the features
    rs_fts = []
    for ft in features:
        rs_fts.append(mean_std_normalize(ft))

    return names, rs_fts
