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


def series2_gen(length, dim):
    # starts off as array of zeros.
    # 0 -> 1 rules:
    # every cell has p=0.1 of switching to 1, when 0.
    # p=0.5/0.9 with one/two neighbours that are 1.
    # 1 -> 0 rules:
    # p=0.2 in general
    # p=0.4/0.6 with one or two of the leftmost neighbours active

    # lacks longer term dependencies. but maybe good enough for now
    series = np.zeros((length, dim))
    for t in range(1, length):
        timestep = np.zeros((dim,))
        for idx in range(dim):
            if series[t-1, idx] == 0:
                odds = 0.1
                if idx > 0 and series[t-1, idx-1] == 1: odds += 0.4
                if idx < dim -1 and series[t-1, idx+1] == 1: odds += 0.4
                if odds > np.random.uniform(): series[t, idx] = 1
            else:
                odds = 0.2
                if idx > 0 and series[t-1, idx-1] == 1: odds += 0.2
                if idx > 1 and series[t-1, idx-2] == 1: odds += 0.2
                if odds < np.random.uniform(): series[t, idx] = 1
    return series


# print series2_gen(30, 17)
def series2_check(series):
    # counts rule applications: 6 rules
    # 0->1:  0) p=0.1  1) p=0.5  2) p=0.9
    # 1->0:  3) p=0.2  4) p=0.4  5) p=0.6
    # choice in dimension 2: 0) picked 0, 1) picked 1)
    score = np.zeros((6, 2), dtype=np.int)
    series = series.round().astype(np.int32)
    length = series.shape[0]
    dim = series.shape[1]
    for t in range(1, length):
        timestep = np.zeros((dim,))
        for idx in range(dim):
            choice = series[t, idx]
            if series[t-1, idx] == 0:
                rule = 0
                if idx > 0 and series[t-1, idx-1] == 1: rule += 1
                if idx < dim - 1 and series[t-1, idx+1] == 1: rule += 1
                score[rule, choice] += 1
            else:
                rule = 3
                if idx > 0 and series[t-1, idx-1] == 1: rule += 1
                if idx > 1 and series[t-1, idx-2] == 1: rule += 1
                score[rule, choice] += 1

    # return measured probabilities of 0 in each case. should be around [0.9 0.5 0.1  0.2 0.4 0.6]
    return score[:, 0].astype(np.float32) / np.sum(score, 1).astype(np.float32)

s = series2_gen(200, 100)
print(s)
c = series2_check(s)
print(c)