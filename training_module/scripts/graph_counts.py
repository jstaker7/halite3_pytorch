import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


with open('/path/move_counts.pkl', 'rb') as handle:
    counts = pickle.load(handle)

print(counts.keys())

# Delete bad games (games that ended early)
del counts['1_32_335']
del counts['1_32_346']

outpath = '/path/count_graphs'

which = ['o', 'n', 'e', 's', 'w', 'c']
for k, v in counts.items():
    for i in range(v.shape[0]):
        plt.scatter(list(range(v.shape[1])), v[i], label=which[i])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, '{}.png'.format(k)))

    plt.clf()

    maxed = np.max(v, 0)
    #maxed[maxed < 1e-32] = 1e-32

    #print((1 - v) / still)
    copied = v.copy()
    copied[copied < 1e-32] = 1e-32
    copied = 1/copied
    copied[v < 1e-32] = 0

    copied = copied * maxed


# TODO: Convert the values into weights (are they already in a usable format?)
