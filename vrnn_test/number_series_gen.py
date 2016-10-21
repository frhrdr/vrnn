import numpy as np


def save_series(number, length, dim, file_path, sid, noise=True, level=0.5):
    s_list = [series1_gen, series2_gen, series3_gen,
              series4_gen, series5_gen, series6_gen,
              series7_gen]
    series = np.ndarray((length, number, dim))
    for idx in range(number):
        si = s_list[sid-1](length, dim)
        if noise:
            si = add_noise(si, noise_stdev=level)
        series[:, idx, :] = si
    np.save(file_path, series)


def load_series(file_path):
    return np.load(file_path)


def series_check(series, sid):
    c_list = [series1_check, series2_check, series3_check,
              series4_check, series5_check, series6_check,
              series7_check]
    c_list[sid-1](series)

# integers 0 - 127
# a_n :=
# 50%:  a_(n-1) - 9 mod 128
# 30%:  a_(n-1) + a_(n-3) mod 128
# 20%:  a_(n-2) + a_(n-4) mod 128
# starts with 0 0 0 0
# returns np.array
def series1_gen(length, dim):
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
    series = np.minimum(series, np.ones(series.shape))
    series = np.maximum(series, np.zeros(series.shape))
    series = series.round().astype(np.int32)
    length = series.shape[0]
    dim = series.shape[1]
    for t in range(1, length):
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

    norm = np.sum(score, 1).astype(np.float32)
    norm = np.maximum(norm, np.ones(norm.shape))
    # print(length, dim)
    # return measured probabilities of 0 in each case. should be around [0.9 0.5 0.1  0.2 0.4 0.6]
    return score[:, 0].astype(np.float32) / norm


def series3_gen(length, dim, val=5):
    # dumbest one yet. for debugging
    series = np.zeros((length, dim))
    series[1:, :] = - val
    series[1::2, 1::2] = val
    series[2::2, ::2] = val

    return series


def series3_check(series):
    print(np.round(series))
    series[series <= 0] = 0
    series[series > 0] = 1
    print(series)


def series4_gen(length, dim, odds=0.5):
    # version of 3 with probabilisms (TM)
    series = series3_gen(length, dim)
    s = np.zeros(series.shape)
    rand = np.random.uniform(size=(length, dim))
    s[(series <= 0) * (rand <= odds)] = -9
    s[(series <= 0) * (rand > odds)] = 3
    s[(series > 0) * (rand <= odds)] = 9
    s[(series > 0) * (rand > odds)] = -3
    return s


def series4_check(series):
    s = np.zeros(series.shape)
    s[series <= -6] = 7
    s[(series <= 6) * (series > 0)] = 1
    s[series > 6] = 8
    s[(series > -6) * (series <= 0)] = 0
    print([np.sum(s == 0), np.sum(s == 1), np.sum(s == 8), np.sum(s == 7)])
    print(s)


def series5_gen(length, dim):
    # deterministic spin on s2
    # on a circle, both closest neighbours are 1 -> 0
    # barring this, if 3/6 rightmost neighbours are 1 -> 1
    s = np.zeros((length, dim))
    s[0, :] = (np.random.uniform(size=(dim,)) > 0.5)

    for ti in range(1, length):
        for di in range(dim):
            if s[ti-1, di] == 1 and sum([s[ti-1, k % dim] for k in range(di-1, di+2)]) >= 3:
                s[ti, di] = 0
            elif s[ti-1, di] == 0 and sum([s[ti-1, k % dim] for k in range(di+1, di+7)]) >= 3:
                s[ti, di] = 1
            else:
                s[ti, di] = s[ti-1, di]
    return s


def series5_check(series):
    s = (series > 0.5)
    dim = s.shape[1]
    e = np.zeros((3,))  #0: 1 -> 0 rule ignored  1: 0 -> 1 rule ignored  2: random change
    c = np.zeros((3,))
    for ti in range(1, s.shape[0]):
        for di in range(dim):
            if s[ti-1, di] == 1 and sum([s[ti-1, k % dim] for k in range(di-1, di+2)]) >= 3:
                c[0] += 1
                if s[ti, di] == 1:
                    e[0] += 1
            elif s[ti-1, di] == 0 and sum([s[ti-1, k % dim] for k in range(di+1, di+7)]) >= 3:
                c[1] += 1
                if s[ti, di] == 0:
                    e[1] += 1
            else:
                c[2] += 1
                if s[ti, di] != s[ti-1, di]:
                    e[2] += 1
    print(e)
    print(c)
    return e, c


def series6_gen(length, dim, val=5):
    # spin on 3 with 0 fillers to test deeper memory
    series = np.zeros((length, dim))
    series[1:, :] = - val
    series[1::4, 1::2] = val
    series[3::4, ::2] = val
    return series


def series6_check(series):
    s = np.zeros(series.shape)
    s[series <= -2] = 0
    s[series > 2] = 1
    s[(series > -2) * (series <= 2)] = 8
    print(s)


def series7_gen(length, dim, odds=0.5):
    # version of 4 that's not conflicting
    series = series3_gen(length, dim)
    s = np.zeros(series.shape)
    rand = np.random.uniform(size=(length, dim))
    s[(series <= 0) * (rand <= odds)] = 9
    s[(series <= 0) * (rand > odds)] = 3
    s[(series > 0) * (rand <= odds)] = -9
    s[(series > 0) * (rand > odds)] = -3
    return s


def series7_check(series):
    s = np.zeros(series.shape)
    s[series > 6] = 9
    s[(series <= 6) * (series > 0)] = 3
    s[(series > -6) * (series <= 0)] = -3
    s[series <= -6] = -9
    print([np.sum(s == -9), np.sum(s == -3), np.sum(s == 3), np.sum(s == 9)])
    print(s)
    print(np.round(series))


def add_noise(series, noise_stdev):
    return series + np.random.normal(scale=noise_stdev, size=series.shape)
