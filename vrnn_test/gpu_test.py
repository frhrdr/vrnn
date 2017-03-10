import numpy as np

a = np.load('data/logs/handwriting_15/params.pkl')
for key in a:
    print(key, ': ', a[key])
