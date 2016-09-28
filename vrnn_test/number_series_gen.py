import numpy as np


def save_series(function, number, length, file_path):
    series = np.ndarray((number, length))
    for idx in range(number):
        series[idx, :] = function(length)
    np.save(file_path, series)


def load_series(file_path):
    return np.load(file_path)


# integers 0 - 127
# a_n :=
# 50%:  a_(n-1) - 9 mod 128
# 30%:  a_(n-1) + a_(n-3) mod 128
# 20%:  a_(n-2) + a_(n-4) mod 128
# starts with 0 0 0 0
# returns np.array
def series1_gen(length):
    s = np.ndarray((length,), dtype=np.int16)
    s[:4] = 0
    for idx in range(4, length):
        switch = np.random.randint(0, 9)
        if switch < 5:
            s[idx] = (s[idx-1] - 9) % 128
        elif switch < 8:
            s[idx] = (s[idx-1] + s[idx-3]) % 128
        else:
            s[idx] = (s[idx-2] + s[idx-4]) % 128

    return s


def series1_check(series):
    # counters: 0 - init, 1 - rule 1, 2 - rule 2, 3 - rule 3, 4 - error
    # biased towards rules 1 and 2, when resolving ambiguities
    score = np.zeros(5)
    for idx in range(4):
        if series[idx] == 0:
            score[0] += 1
        else:
            score[4] += 1
    for idx in range(4, series.size):
        if series[idx] == (series[idx-1] - 9) % 128:
            score[1] += 1
        elif series[idx] == (series[idx-1] + series[idx-3]) % 128:
            score[2] += 1
        elif series[idx] == (series[idx-2] + series[idx-4]) % 128:
            score[3] += 1
        else:
            score[4] += 1

    return score / np.sum(score)

path = 'data/series1_1000_by_300.npy'
save_series(series1_gen, 1000, 300, path)
# s = load_series(path)
# print(s)
